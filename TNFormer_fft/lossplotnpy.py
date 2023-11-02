import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# # 设置中文字体
# font = FontProperties(fname='宋体', size=10)  # 替换为你的中文字体文件和字号
#
#
# # 创建一个1x3的子图
# fig, axs = plt.subplots(1, 3, figsize=(12, 4))
#
# # 被试列表
# subjects = ['1', '2', '3']
# accuracy = [0.853, 0.837, 0.378]
#
# # 迭代处理每个被试的损失数据
# for i, subject in enumerate(subjects):
#     # 读取训练集、验证集和测试集的损失数据
#     train_loss = np.load(f'losses_train{subject}.npy')
#     val_loss = np.load(f'losses_vali{subject}.npy')
#     test_loss = np.load(f'losses_loss{subject}.npy')
#
#     # 创建横坐标轴
#     epochs = np.arange(len(train_loss))
#
#     # 绘制损失曲线
#     axs[i].plot(epochs, train_loss, label='Train Loss')
#     axs[i].plot(epochs, val_loss, label='Validation Loss')
#     axs[i].plot(epochs, test_loss, label='Test Loss')
#     axs[i].set_title(f'Subject {i + 1}')
#
#     # 添加识别正确率文本
#     accuracy_text = f'Accuracy: {accuracy[i]:.2f}%'
#     axs[i].text(0.5, 0.7, accuracy_text, transform=axs[i].transAxes)
#
#     # 设置共享的y轴标签
#     axs[i].set_ylabel('Loss')
#     axs[i].set_xlabel('Epoch')
#     # 添加图例
#     axs[i].legend()
#
#
# # 添加文字说明
# fig.suptitle('14-class, 0.5s, epoch=300, batch_size=20, earlystop_patience=30, learning_rate=0.001\n'
#              'subject 3 can be improved by increasing learning rate')
# # fig.suptitle('subject 3 increase learning rate')
# # Add a text description below the subplot
# description = ""
# plt.figtext(0.5, 0.01, description, ha="center")
# # 调整子图之间的间距
# plt.tight_layout()
# plt.show()
#



num_subjects = 1
num_subjects = np.append(num_subjects,np.arange(1,14))



# 循环处理每个被试的数据
for subject in num_subjects:
    # 创建1x3的子图布局
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # 读取损失数据和准确率数据
    train_loss = np.load(f'losses_train{subject}.npy')
    val_loss = np.load(f'losses_vali{subject}.npy')
    test_loss = np.load(f'losses_test{subject}.npy')

    val_acc = np.load(f'accuracy_vali{subject}.npy')
    test_acc = np.load(f'accuracy_test{subject}.npy')

    # 绘制损失曲线
    axs[0].plot(train_loss, label=f'Subject {subject} Train Loss')
    axs[0].plot(val_loss, label=f'Subject {subject} Validation Loss')
    axs[0].plot(test_loss, label=f'Subject {subject} Test Loss')
    # axs[0].set_title('Loss')
    axs[0].legend()
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    # 绘制准确率曲线
    axs[1].plot(val_acc, label=f'Subject {subject} Validation Accuracy')
    axs[1].plot(test_acc, label=f'Subject {subject} Test Accuracy')
    # axs[1].set_title('Accuracy')
    axs[1].legend()
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
# 显示图形
fig.suptitle('Subject' + str(subject))
plt.tight_layout()
plt.show()