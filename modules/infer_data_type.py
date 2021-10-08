"""
Infer Data Type

Data representation can typically be classified into 3 categories:
1) Vector
2) Matrix data (Images)
3) Sequences
"""

import numpy as np

class InferDataType(object):
	"""This class will return the type of the data."""
	def getType(self, data):
		# Determine if it's Pandas or Numpy
		if("pandas" in str(type(data))):
			data = data.as_matrix()
		
		# Determine if each sample is a vector
		if(data.ndim == 2):
			# Then it's either vector or a sequence
			# Check the type of the first item in the first row
			# If it's a list then our datum is a sequence
			# If it's not a list then it's a vector
			if(type(data[0][0]) is list):
				self.type = "Sequence"
				return self.type
			else:
				self.type = "Vector"
				return self.type
		# Then is it an image?
		elif(data.ndim == 4):
			# It is an image
			self.type = "Image"
			return self.type
		# Unknown
		else:
			self.type = "Unknown"
			return self.type

	def __init__(self):
		self.type = ""
