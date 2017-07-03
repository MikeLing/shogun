#!/usr/bin/env python
parameter_list=[[]]

def labels_io():
	from shogun import RegressionLabels, CSVFile
	lab=RegressionLabels()
	f=CSVFile("../data/label_train_regression.dat","r")
	f.set_delimiter(" ")
	lab.load(f)
	#print lab.get_labels()
	return lab

if __name__=='__main__':
	print('Labels IO')
	labels_io(*parameter_list[0])
