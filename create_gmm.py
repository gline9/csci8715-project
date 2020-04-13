from sklearn.mixture import GaussianMixture as GMM
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump


routes = pd.read_pickle("routes.pkl")[['work_lat', 'work_lon', 'home_lat', 'home_lon']]
gmm_20 = GMM(20, covariance_type='full', random_state=0).fit(routes.values)
dump(gmm_20, '20_gmm.joblib')

# r = np.arange(1, 31, 2)
# gmms = []
# million_values = routes.sample(1_000_000).values
# for num_gaussians in r:
#     print('fitting gaussian ' + str(num_gaussians))
#     gmms.append(GMM(num_gaussians, covariance_type='full', random_state=0).fit(routes.sample(100000).values))

# aics = []
# for num_gaussians in range(0, len(gmms)):
#     print('calculating aic for gaussian #' + str(num_gaussians + 1))
#     aics.append(gmms[num_gaussians].aic(million_values))

# plt.plot(r, aics);
# plt.show()

# print(gmm.sample(1))

