import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/home/peter/code/projects/tradesys')

from tradesys.ml import BaseMLModel 

class MLModel(BaseMLModel):
    def get_response(self):
        an = self.model.predict(self.x)
        an = an[0]
        if an == 0:
            return 1
        elif an == 1:
            return 0
        elif an == 2:
            return -1

