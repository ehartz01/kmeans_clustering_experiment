#pp3 Ethan Hartzell
#kmeans algorithm
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from math import log, exp
import random
import glob

#reads in and stores info about a .arff file
class arff_file:
    def __init__(self,filename):
		self.relation = ""		#relation name
		self.attributes = []	#feature names
		self.classes = []
		self.data = []	
		self.labeled_data = []		#examples
		self.file = filename
		f = open(filename)		#open the file
		tmp = f.readlines()
		data = False
		#store the relevent parts of the file
		for line in tmp:		#for each line
		    ls = line.split()	#for each word
		    for i, word in enumerate(ls):
				if data:
					if line.upper() != "@DATA":
						self.data.append(line.split(",")[:-1])	#append all but last word
						self.labeled_data.append(line.split())
					continue
				if word.upper() == "@Relation":
					self.relation = ls[i+1]	#relation gets next word
					continue
				if word.upper() == "@ATTRIBUTE":
					if "class" not in line:
						self.attributes.append(line)	#save the whole line for the attribute
					elif "class" in line.split():
						for c in line.split(","):
							self.classes.append(c)
				if word.upper() == "@DATA":
					data = True
					continue
		f.close()

    def get_data(self):
        return self.data
    def get_labeled_data(self):
    	return self.labeled_data
    def replace_data(self,nd):
        self.data = nd

#implements the clustering algorithm, stores means and clusters and performs calculations on them
class kmeans_cluster:
	#reads in the dataset to be clustered along with the K
	def __init__(self, dset, k):
		self.num_features = len(dset.attributes)
		self.data = dset.get_data()	#should be an array of arrays, each one containing feature values for an example
		self.k = k
		self.initial_means = []
		self.clusters = defaultdict(list)
	#initializes the means as random examples
	def random_means(self):
		examples = self.data[:]
		self.initial_means = []
		for initial_mean in range(0,self.k):
			val = examples.pop(random.randint(0,len(examples)-1))
			tmp = []
			for v in val:
				tmp.append(float(v))
			self.initial_means.append(tmp)
	#smart initialization of the means
	def smart_init(self):
		#first start with random means
		examples = self.data[:]
		self.initial_means = []
		means = []
		#pop a random one
		val = examples.pop(random.randint(0,len(examples)-1))
		tmp = []
		for v in val:
			tmp.append(float(v))
		means.append(tmp)
		#then pick ten more random unused examples
		for m in range(0,self.k - 1):
			ten_more = []
			for example in range(0,10):
				val = examples.pop(random.randint(0,len(examples)-1) )
				tmp = []
				for v in val:
					tmp.append(float(v))
				ten_more.append(tmp)
			#among these ten, select the one that is furthest away from all previously selected means
			distances = []
			for candidate in ten_more:
				mindists= []
				for mean in means:
					mindists.append(np.linalg.norm(np.array(candidate)-np.array(mean)))
				distances.append(min(mindists))
			to_app = ten_more.pop(np.argmax(distances))
			means.append(to_app)
			for x in ten_more:
				examples.append(x)
			#examples.remove(to_app)
		self.initial_means = means

	#produces k clusters in the form of a dictionary of lists
	def get_clusters(self):
		self.clusters = defaultdict(list)
		prev_clusters = defaultdict(list)
		prev_means = []
		converged = False
		counter = 0
		while (converged != True):
			counter += 1
			clusters = defaultdict(list)
			for example in self.data:
				distances = []
				for mean in self.initial_means:
					#calculate the distance to each mean and associate the example with the closest one
					distance = 0.0
					floatexample = []
					for feat in example:
						floatexample.append(float(feat))
					distance = np.linalg.norm(np.array(floatexample)-np.array(mean))
					distances.append(distance)
				clusters[np.argmin(distances)].append(example)
			#calculate new means based on those clusters
			new_means = range(0,len(self.initial_means))
			prev_means = self.initial_means[:]
			for cluster_id in clusters.keys():
				new_mean = []
				for i in range(0,self.num_features):
					feature_mean = 0.0
					count = 0.0
					for example in clusters[cluster_id]:
						count +=1
						feature_mean += float(example[i])
					feature_mean /= count
					new_mean.append(feature_mean)
				self.initial_means[cluster_id] = new_mean
			#test if converged
			if prev_means == self.initial_means:
				self.clusters = clusters
				return clusters
			else:
				prev_clusters = clusters
	#returns the cluster scatter
	def cluster_scatter(self):
		cluster_scatters = 0.0
		for cluster in self.clusters.keys():
			cluster_scatter = 0.0
			for example in self.clusters[cluster]:
				dist_to_mean = 0.0
				floatexample = []
				for feat in example:
					floatexample.append(float(feat))
				dist_to_mean = np.linalg.norm(np.array(floatexample)-np.array(self.initial_means[cluster]))
				cluster_scatter += dist_to_mean
			cluster_scatters += cluster_scatter
		return cluster_scatters
	#calculates NMI between the clustered data and another cluster both in dictionary form
	def NMI(self,labeled_data):
		correct_clusters = self.lab_to_clus(labeled_data)
		class_entropy = self.entropy(correct_clusters)
		cluster_entropy = self.entropy(self.clusters)
		conditional_entropy = 0.0
		total_c = 0.0
		for cluster in self.clusters.keys():
			total_c += len(self.clusters[cluster])
		for cluster in self.clusters.keys():
			negprobcluster = -1*(len(self.clusters[cluster])/total_c)
			ent_sum = 0.0
			for y in correct_clusters.keys():
				total_y = 0.0
				for ex in correct_clusters[y]:
					if ex in self.clusters[cluster]:
						total_y += 1.0
				if total_y != 0:
					ent_sum += (total_y/len(self.clusters[cluster]))*(log(total_y)-log(len(self.clusters[cluster])))
			conditional_entropy += negprobcluster*ent_sum
			#print conditional_entropy
		iyc = class_entropy - conditional_entropy
		nmi = (2*iyc)/(class_entropy+cluster_entropy)
		return nmi
	#calculates and returns the entropy of a cluster (in the form of a dictionary)
	def entropy(self,clusters):
		total = 0.0
		entropy = 0.0
		for label in clusters.keys():
			total += len(clusters[label])
		for label in clusters.keys():
			val = (len(clusters[label])/total)*log(len(clusters[label])/total)
			entropy += val
		entropy *= -1
		return entropy
	#turns the labeled dataset of .arff files into a form usable for our NMI function
	def lab_to_clus(self,labeled_data):	#each example in the labeled data is an array with one string
		clusters = defaultdict(list)
		for example in labeled_data:
			ex = example[0].split(",")
			key = ex.pop(-1)
			clusters[key].append(ex)
		return clusters
#runs the clustering algorithm on a data set and gets its cluster scatter and nmi with the true labeled data
def random_run(dset):
	x = kmeans_cluster(dset,len(dset.classes))
	x.random_means()
	c = x.get_clusters()
	cs = x.cluster_scatter()
	nmi = x.NMI(dset.get_labeled_data())
	return (cs,nmi)
#same as above but with smart initialization
def smart_run(dset):
	x = kmeans_cluster(dset,len(dset.classes))
	x.smart_init()
	c = x.get_clusters()
	cs = x.cluster_scatter()
	nmi = x.NMI(dset.get_labeled_data())
	return (cs,nmi)

#for each .arff file in this folder, run 10 random runs and one smart init and plot results
for file in glob.glob("*.arff"):
	exp_res = []
	print file
	dset = arff_file(file)
	for i in range(0,10):
		exp_res.append( random_run(dset) )
	exp_res.append( smart_run(dset) )
	y1 = []
	y2 = []
	for cs,nmi in exp_res:
		y1.append(cs)
		y2.append(nmi)
	x = range(1,len(exp_res)+1)
	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax1.bar(x,y1)
	ax1.set_ylabel("CS")
	plt.ylim((min(y1)-20,max(y1)+20))
	ax2 = fig.add_subplot(212)
	ax2.bar(x,y2)
	ax2.set_ylabel('NMI', color='r')
	plt.ylim((min(y2)-.1,max(y2)+.1))
	plt.title(file)
	plt.show()

#cluster the data with different k values and collect cluster scatter values
def k_experiment(dset):
	results = []
	for k in range(2,23):
		res = []
		x = kmeans_cluster(dset,k)		#do 10 random inits and pick the best one
		for t in range(0,10):
			x.random_means()
			c = x.get_clusters()
			cs = x.cluster_scatter()
			res.append(cs)
		results.append(min(res))
	return results
#run each data file through the k experiment and plot the results
for file in glob.glob("*.arff"):
	print file
	dset = arff_file(file)
	res = k_experiment(dset)
	y1 = res
	x = range(2,23)
	plt.figure()
	plt.plot(x,y1)
	plt.xlabel("k")
	plt.ylabel("CS")
	plt.title(file)
	plt.show()