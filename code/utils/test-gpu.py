
import tensorflow as tf

def main():
	physical_devices=tf.config.list_physical_devices('GPU')
	if(len(physical_devices)>0):
		print("-->",physical_devices[0])
		print(tf.__version__)
		tf.config.experimental.set_memory_growth(physical_devices[0],True)
	else:
		print("no gpu")
	return

main()
