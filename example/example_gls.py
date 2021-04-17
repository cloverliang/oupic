# +
import dendropy as dp
import numpy as np
import pandas as pd
from scipy import stats

from oupic.ou_calculator import OUCalculator
from oupic.pic import PIC
from oupic.gls import BMcov, OUcov, gls_fit
# -

filename = '/Users/cong/Workspace/repos/pylce/data/brawand_tree.nwk'
tree = dp.Tree.get_from_path(filename, schema='newick')

# generate sample data
X_dic = {}
Y_dic = {}
i = 0
for taxon in tree.taxon_namespace:
    X_dic[taxon.label] = i
    Y_dic[taxon.label] = X_dic[taxon.label] + 3*np.random.rand()
    i += 1

# +
# GLS fit
X = np.array(list(X_dic.values()))
Y = np.array(list(Y_dic.values()))
my_fit = gls_fit(X, Y, OUcov(tree, 0.5), add_intercept=True)

# format data
RSS = my_fit.pop('RSS')
my_fit = pd.DataFrame(my_fit)
my_fit = my_fit.rename(index={0:'intercept', 1:'slope'})

# print
format_str = "=" * 20 + " %s " + "=" * 20
print('\n' + format_str % 'GLS fit')
print(my_fit)
print('\nRSS: ', RSS)

# +
# Calculate PIC and OLS fit

# Initiate calculator
calculator = OUCalculator(ts_lambda=0.5)
# Calcualte PIC
pic_X = PIC(tree, calculator, X_dic)
pic_Y = PIC(tree, calculator, Y_dic)
pic_X.calc_contrast()
pic_Y.calc_contrast()
X_pic = pic_X.contrasts.loc['contrast_standardized'].astype(float)
Y_pic = pic_Y.contrasts.loc['contrast_standardized'].astype(float)

# OLS fit
my_ols = gls_fit(X_pic, Y_pic, np.identity(len(Y_pic)), add_intercept=False)

# print
print('\n' + format_str % 'OLS fit on PICs')
[ print(key,':',value) for key, value in my_ols.items() ]

# +
# Correlation test between PICs
m = len(X_pic)
rval = np.corrcoef(X_pic.astype(float), Y_pic.astype(float))[0,1]
tval = rval * np.sqrt((m - 2)/(1 - rval**2))
pval = 1 - np.abs(1 - 2*stats.t.cdf(tval, df=m-2))

# print 
print('\n' + format_str % 'Correlation test between PICs')
print('Correlation Coefficient: ', rval)
print('R-squared: ', rval**2)
print('Degree of freedom: ', m-2)
print('Test p value: ', pval)
# -

my_ols_icpt = gls_fit(X_pic, Y_pic, np.identity(len(Y_pic)), add_intercept=True)
[print(key, ':', value) for key, value in my_ols_icpt.items()]


