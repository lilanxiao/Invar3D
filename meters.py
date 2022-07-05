import random

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':.3f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.prefix = prefix
        self.meters = {}
    
    def update(self, batch, dic):
        if batch == 0:
            for key in dic:
                m = AverageMeter(key, ":.4f")
                self.meters[key] = m
        for key in dic:
            self.meters[key].update(dic[key])

    def display(self, batch):
        meters = set(self.meters.values())
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def data_gen():
    ret = {}
    ret["data1"] = random.random()
    ret["sine"] = random.random() * 2
    ret["adfasdf"] = random.random() * 3
    return ret


if __name__ == "__main__":
    BATCH = 1000
    meters = {}
    progress = ProgressMeter(BATCH)
    for i in range(BATCH):
        ret = data_gen()
        
        progress.update(i, ret)
        progress.display(i)
    
    