import dendropy as dp
import pandas as pd

from oupic.ou_calculator import OUCalculator
from oupic.pic import PIC


def test_pic_with_ou_calculator(tree: dp.Tree) -> None:
    calc = OUCalculator(0.5)
    taxa_val = {}
    i = 1
    for taxon in tree.taxon_namespace:
        taxa_val[taxon.label] = i
        i += 1

    pic = PIC(tree, calc, taxa_val)
    pic.calc_contrast()

    node_coef = pd.DataFrame(
        {
            "Internal_1": {"A": 0.6045901829462685, "B": 0.6045901829462685, "C": 0.0},
            "Internal_2": {
                "A": 0.4457494222185535,
                "B": 0.4457494222185535,
                "C": 0.5656299054048656,
            },
        }
    )
    contrast_coef = pd.DataFrame(
        {
            "Internal_1": {"A": 1.257766554997121, "B": -1.257766554997121, "C": 0.0},
            "Internal_2": {
                "A": 0.5948021943740204,
                "B": 0.5948021943740204,
                "C": -1.1896043887480405,
            },
        }
    )
    contrasts = pd.DataFrame(
        {
            "Internal_1": {
                "is_contrast": True,
                "contrast_standardized": -1.257766554997121,
                "dist_to_parent": 1.6201145069582776,
                "nd_value": 1.8137705488388056,
            },
            "Internal_2": {
                "is_contrast": True,
                "contrast_standardized": -1.7844065831220604,
                "dist_to_parent": 1.247064220585906,
                "nd_value": 3.0341379828702575,
            },
        }
    )
    assert node_coef.equals(pic.node_coef)
    assert contrast_coef.equals(pic.contrast_coef)
    assert contrasts.equals(pic.contrasts)
