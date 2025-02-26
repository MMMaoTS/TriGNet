from dataset.data_utils import prepare_data_mitpsg, prepare_data_ae, prepare_data_mitpsgbi
import config


def prepare_data():
    if config.DATASET == "mitpsg":
        return prepare_data_mitpsg()
    elif config.DATASET == "mitbi":
        return prepare_data_mitpsgbi()
    elif config.DATASET == "ae":
        return prepare_data_ae()
    else:
        raise ValueError("Invalid dataset")
