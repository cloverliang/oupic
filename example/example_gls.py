# +
import dendropy as dp
import numpy as np
import pandas as pd
import pickle
from munch import Munch
from scipy import stats
from pathlib import Path

from oupic.ou_calculator import OUCalculator
from oupic.pic import PIC
from oupic.gls import GLSFit
from oupic.phylo_cov import PhyloCov

# +
# resolve file path
DIR_NAME = Path(__file__).resolve().parents[1]

# load tree
filename = DIR_NAME / "data" / "acer_tree.nwk"
tree = dp.Tree.get_from_path(filename, schema="newick")

# load simulated data
traitX_filename = DIR_NAME / 'data' / 'simulated_acer_tree_traitX.pickel'
traitY_filename = DIR_NAME / 'data' / 'simulated_acer_tree_traitY.pickel'
with open(traitX_filename, 'rb') as file:
    X_dic = pickle.load(file)
with open(traitY_filename, 'rb') as file:
    Y_dic = pickle.load(file)
    
# load attributes
attr_filename = DIR_NAME / 'data' / 'simulated_acer_tree_attributes.pickel'
with open(attr_filename, 'rb') as file:
    attr_xy = pickle.load(file)
# -

print('Example tree: Acer Tree')
print('Number of taxa: ', len(tree.taxon_namespace))
# tree.print_plot()
print('Two traits were simulated with correlated evolution parameter: ', attr_xy['gamma_xy'])

# +
# ============ Test 1: GLS fit ===============
X = np.array(list(X_dic.values()))
Y = np.array(list(Y_dic.values()))

# covariance matrix
tree_cov = PhyloCov(tree)
ts_sigma = tree_cov.get_cov_mat(attr_xy['lambda_x'])

# fit
my_fit = GLSFit(X, Y, ts_sigma, add_intercept=True)

# format data
R_squared = my_fit.pop('R-squared')
my_fit = pd.DataFrame(my_fit)
my_fit = my_fit.rename(index={0:'intercept', 1:'slope'})

# print
format_str = "=" * 20 + " %s " + "=" * 20
print('\n' + format_str % 'Test 1: GLS fit on raw trait values')
print(my_fit)
print('\nR-squared: %.6f' % R_squared)

# +
# =========== Test 2: OLS fit on PIC (no intercept) ============
# Initiate PIC calculator
calculator = OUCalculator(attr_xy['lambda_x'])
# Calcualte PIC
pic_X = PIC(tree, calculator, X_dic)
pic_Y = PIC(tree, calculator, Y_dic)
pic_X.calc_contrast()
pic_Y.calc_contrast()
X_pic = pic_X.contrasts.loc['contrast_standardized'].astype(float)
Y_pic = pic_Y.contrasts.loc['contrast_standardized'].astype(float)

# OLS fit
ts_sigma = np.identity(len(Y_pic))
my_ols = GLSFit(X_pic, Y_pic, ts_sigma, add_intercept=False)

# print
print('\n' + format_str % 'Test 2: OLS fit on PICs (no intercept)')
for key, value in my_ols.items():
    print(key, ' : %.6f' % value)

# +
# =========== Test 3: Correlation test between PICs ===========
m = len(X_pic)
rval = np.corrcoef(X_pic.astype(float), Y_pic.astype(float))[0,1]
tval = rval * np.sqrt((m - 2)/(1 - rval**2))
pval = 1 - np.abs(1 - 2*stats.t.cdf(tval, df=m-2))

# print 
print('\n' + format_str % 'Test 3: Correlation test between PICs')
print('Correlation Coefficient: %.6f' % rval)
print('R-squared: %.6f' % rval**2)
print('Test p value: %.6f' % pval)

# +
# ============ Test 4: OLS fit on PIC (with intercept) ============
ts_sigma = np.identity(len(Y_pic))
my_ols_icpt = GLSFit(X_pic, Y_pic, ts_sigma, add_intercept=True)

R_squared = my_ols_icpt.pop('R-squared')
my_ols_icpt = pd.DataFrame(my_ols_icpt)
my_ols_icpt = my_ols_icpt.rename(index={0:'intercept', 1:'slope'})

print('\n' + format_str % 'Test 4: OLS fit on PICs (with intercept)')
print(my_ols_icpt)
print('\nR-squared: %.6f' % R_squared)
# -


