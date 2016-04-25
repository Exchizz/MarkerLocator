import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import numpy as np
image = plt.imread("background_with_drones_video10_two_drones.png")

data6 = np.genfromtxt('bagfile_positionPuplishe6.csv', delimiter=',', skip_header=1, skip_footer=0, names=['quality', 'x', 'y'])
data4 = np.genfromtxt('bagfile_positionPuplishe4.csv', delimiter=',', skip_header=1, skip_footer=0, names=['quality', 'x', 'y'])

plot6 = plt.scatter(x=data6['x'], y=data6['y'],color='b')
plot4 = plt.scatter(x=data4['x'], y=data4['y'],color='r')

plt.gca().set_xlabel("Width [px]", fontsize=10)
plt.gca().set_ylabel("Heigth [px]", fontsize=10)

plt.legend([plot4, plot6], ['Marker order 4', 'Marker order 6'], fontsize=10)

plt.imshow(image)
#plt.show()
#plt.savefig('test.png')
plt.savefig('test.eps', bbox_inches='tight')
