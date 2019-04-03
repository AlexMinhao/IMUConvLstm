import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import os
from definitions import *
from import_dataset import *
from utils import *

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 24, 113)
    return out



img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                r'OPPORTUNITY\OppSegBySubjectGesturesFull_113Validation.data')
Opp = OPPORTUNITY(path)
X_train, y_train, X_validation, y_validation, X_test, y_test = Opp.load()  # load_dataset(dp)
assert NB_SENSOR_CHANNELS_113 == X_train.shape[2]

print(" ..after sliding window (training): inputs {0}, targets {1}".format(X_train.shape, y_train.shape))
print(" ..after sliding window (validation): inputs {0}, targets {1}".format(X_validation.shape,
                                                                                     y_validation.shape))
print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))


X_train = X_train.reshape(
            (-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS_113))  # inputs (46495, 1, 24, 113), targets (46495,)
X_train.astype(np.float32), y_train.reshape(len(y_train)).astype(np.uint8)

# Choose label
label_index = np.where(y_train == 10)
X_train = X_train[label_index].reshape((-1, 1, 24, 113))
y_train = y_train[label_index].reshape((-1, 1))

training_set = []
X_train = list(X_train)
y_train = list(y_train)

for i in range(len(y_train)):
    x = X_train[i]
    y = y_train[i]
    xy = (x, y)
    training_set.append(xy)

batch_size = 123
num_epoch = 500
z_dimension = 100  # noise dimension

dataloader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)

print('Dataloader finished !')
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # batch, 32, 28, 28
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(3, stride=3),  # batch, 32, 14, 14
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),  # batch, 64, 14, 14
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 64, 7, 7
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*18, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        x: batch, width, height, channel=1
        '''
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class generator(nn.Module):
    def __init__(self, input_size, num_feature):
        super(generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)  # batch, 3136=1x56x56
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 50, 5, stride=1),  # batch, 50, 56, 56
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1),  # batch, 25, 56, 56
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 1, 3, stride=1),  # batch, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        out = self.fc(x)
        out = out.view(out.size(0), 1, 32, 121)
        out = self.br(out)
        out = self.downsample1(out)
        out = self.downsample2(out)
        out = self.downsample3(out)
        return out
print('Dataloader finished !')




D = discriminator().cuda()  # discriminator model
G = generator(z_dimension, 3872).cuda()  # generator model

criterion = nn.BCELoss()  # binary cross entropy

d_optimizer = torch.optim.Adam(D.parameters(), lr=BASE_lr)
g_optimizer = torch.optim.Adam(G.parameters(), lr=BASE_lr)
print('Dataloader finished !3')
# train
for epoch in range(num_epoch):
    adjust_learning_rate(g_optimizer, epoch)
    adjust_learning_rate(d_optimizer, epoch)
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)
        # =================train discriminator
        real_img = get_variable(img.float())
        real_label = Variable(torch.ones(num_img)).cuda()
        fake_label = Variable(torch.zeros(num_img)).cuda()

        # compute loss of real_img
        real_out = D(real_img)
        d_loss_real = criterion(real_out, real_label)
        real_scores = real_out  # closer to 1 means better

        # compute loss of fake_img
        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = G(z)
        fake_out = D(fake_img)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out  # closer to 0 means better

        # bp and optimize
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ===============train generator
        # compute loss of fake_img
        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = G(z)
        output = D(fake_img)
        g_loss = criterion(output, real_label)

        # bp and optimize
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i+1) % 1 == 0:
            print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                  'D real: {:.6f}, D fake: {:.6f}'
                  .format(epoch, num_epoch, d_loss.data, g_loss.data,
                          real_scores.data.mean(), fake_scores.data.mean()))
    if epoch == 0:
        real_images = to_img(real_img.cpu().data)
        save_image(real_images, './dc_img/real_images.png')

    if epoch % 50 == 0:
        fake_images = to_img(fake_img.cpu().data)
        save_image(fake_images, './dc_img/fake_images-{}.png'.format(epoch+1))

        g_states = {
            'epoch': epoch + 1,
            'state_dict': G.state_dict(),
            'optimizer': g_optimizer.state_dict(),
        }
        d_states = {
            'epoch': epoch + 1,
            'state_dict': D.state_dict(),
            'optimizer': d_optimizer.state_dict(),
        }
        g_states_path = os.path.join(os.getcwd(), './dc_model', "generator_epoch_{}.pth".format(epoch))
        d_states_path = os.path.join(os.getcwd(), './dc_model', "discriminator_epoch_{}.pth".format(epoch))
        torch.save(g_states, g_states_path)
    #torch.save(d_states, d_states_path)