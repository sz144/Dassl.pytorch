import os.path as osp

from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class PACSDA(DatasetBase):
    """PACS for domain adaptation.
    """
    dataset_dir = ''
    domains = ['art_painting', 'cartoon', 'photo', 'sketch']

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS)
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS)
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS)

        super().__init__(train_x=train_x, train_u=train_u, test=test)

    def _read_data(self, input_domains):
        items = []

        for domain, dname in enumerate(input_domains):
            domain_dir = osp.join(self.dataset_dir, dname)
            class_names = listdir_nohidden(domain_dir)
            class_names.sort()

            for label, class_name in enumerate(class_names):
                class_path = osp.join(domain_dir, class_name)
                imnames = listdir_nohidden(class_path)

                for imname in imnames:
                    impath = osp.join(class_path, imname)
                    item = Datum(impath=impath, label=label, domain=domain)
                    items.append(item)

        return items


@DATASET_REGISTRY.register()
class VLCSDA(DatasetBase):
    """VLCS for domain adaptation
    """
    dataset_dir = ''
    domains = ['CALTECH', 'LABELME', 'PASCAL', 'SUN']

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, 'images')
        self.split_dir = osp.join(self.dataset_dir, 'splits')

        # if not osp.exists(self.dataset_dir):
        #     dst = osp.join(root, 'pacs.zip')
        #     self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        # train = self._read_data(cfg.DATASET.SOURCE_DOMAINS, 'train')
        # val = self._read_data(cfg.DATASET.SOURCE_DOMAINS, 'crossval')
        # test = self._read_data(cfg.DATASET.TARGET_DOMAINS, 'test')
        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split='full')
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS, split='full')
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, split='test')

        super().__init__(train_x=train_x, train_u=train_u, test=test)

    def _read_data(self, input_domains, split):
    # def _read_data(self, input_domains):
        items = []
        # splits = ['full', 'test']

        for domain, dname in enumerate(input_domains):
            domain_dir = osp.join(self.dataset_dir, dname)
            # for split in splits:
            domain_split_dir = osp.join(domain_dir, split)
            class_names = listdir_nohidden(domain_split_dir)
            class_names.sort()

            for label, class_name in enumerate(class_names):
                class_path = osp.join(domain_split_dir, class_name)
                imnames = listdir_nohidden(class_path)

                for imname in imnames:
                    impath = osp.join(class_path, imname)
                    item = Datum(impath=impath, label=label, domain=domain)
                    items.append(item)

        return items
