from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets import play_frame

root_dir = '/home/hp/DataSet/DVS128Gesture/'
train_set = DVS128Gesture(root_dir, train=True, data_type='frame', frames_number=8, split_by='number')
frame, label = train_set[0]
print(frame.shape)
from spikingjelly.datasets import play_frame
frame, label = train_set[400]
print('----frame.shape---')
print(frame)
print('----label---')
print(label)
play_frame(frame)