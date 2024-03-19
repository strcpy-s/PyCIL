import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000
from tqdm import tqdm


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        '''
        以cifar10为例，cifar10含有10个类。虽然datamanager一次性读取了所有类和所有类的数据(_setup_data函数读取)
        但是为了体现增量学习，所以可以把10个类拆成多次任务学习。因此在bic.json里面设置增量参数（例如init_cls和increment），
        init_cls代表机器第一次学习类的个数，self._increments代表每次学习类别个数（increment）的列表, increment代表
        每次学习类别个数。所以init_cls必须小于等于len(_class_order)。最后要满足sum(self._increments)等于10,
        len(self._increments)代表了任务个数
        '''
        assert init_cls <= len(self._class_order), "No enough classes."  # assert为假时才抛异常
        self._increments = [init_cls]  # self._increments代表每次任务所含类别个数列表
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)  # 最后一次学习类别个数可能不到increment

    @property
    # @property这个装饰器，这个装饰器的功能就是把类中的方法(函数)，
    # 变成属性，调用时不加括号
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_accumulate_tasksize(self, task):
        return sum(self._increments[:task + 1])

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
            self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        """
        mode参数用于选择不同的transform
        """
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:  # 此处这里是将样本库里的样本添加进来
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        # np.concatenate(data)才行(降维),否则就是一个五维tensor
        # np.concatenate(targets)才行(降维),否则就是一个二维tensor
        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_finetune_dataset(self, known_classes, total_classes, source, mode, appendent, type="ratio"):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))
        val_data = []
        val_targets = []

        old_num_tot = 0
        appendent_data, appendent_targets = appendent

        for idx in range(0, known_classes):
            append_data, append_targets = self._select(appendent_data, appendent_targets,
                                                       low_range=idx, high_range=idx + 1)
            num = len(append_data)
            if num == 0:
                continue
            old_num_tot += num
            val_data.append(append_data)
            val_targets.append(append_targets)
        if type == "ratio":
            new_num_tot = int(old_num_tot * (total_classes - known_classes) / known_classes)
        elif type == "same":
            new_num_tot = old_num_tot
        else:
            assert 0, "not implemented yet"
        new_num_average = int(new_num_tot / (total_classes - known_classes))
        for idx in range(known_classes, total_classes):
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx + 1)
            val_indx = np.random.choice(len(class_data), new_num_average, replace=False)
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
        val_data = np.concatenate(val_data)
        val_targets = np.concatenate(val_targets)
        return DummyDataset(val_data, val_targets, trsf, self.use_path)

    def get_dataset_with_split(
            self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name)  # 根据数据名称获得数据
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets # 都是np.ndarray
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        '''
        如果用cifar10，则有十个类,
        不做重拍列的话，代码默认class_id=index，即0=0。注意train_targets为包含class_id列表，
        但是train_targets里面不满足class_id=index。
        这么做目的是进一步打乱数据顺序
        '''
        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()  # 打乱编号
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        # 这里是要把实际的类编号映射到order里面的索引。即train_target(注意这里表示一个) = order.index(train_target)
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range) -> np.ndarray | list:
        '''
        返回np.ndarray(如果x是np.ndarray) | list(x不是np.ndarray)
        '''
        # 从target(也就是y变量)列表（已被映射成order的坐标）中选取target列表的下标, 注意np.where返回的是元组
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]

        if isinstance(x, np.ndarray):
            x_return = x[idxes]
        else:
            x_return = []
            for id in idxes:
                x_return.append(x[id])
        return x_return, y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
