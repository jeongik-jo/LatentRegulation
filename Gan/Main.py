import time
import HyperParameters as HP
import Data
from . import Evaluate, Model, Train


def train_gan():
    generator = Model.Generator()
    discriminator = Model.Discriminator()
    train_dataset, test_dataset = Data.load_dataset()

    fids = []
    for epoch in range(HP.epochs):
        print('iter', epoch)
        start = time.time()
        Train.train(generator.model, discriminator.model, train_dataset)

        print('saving...')
        generator.save_images(epoch)
        discriminator.save()
        generator.save()
        print('saved')

        if HP.evaluate_model and (epoch + 1) % HP.epochs_per_evaluate == 0:
            fid = Evaluate.get_fid(generator.model, test_dataset)
            print('fid :', fid)
            fids.append(fid)

        print('time: ', time.time() - start)

    if not HP.evaluate_model:
        fid = Evaluate.get_fid(generator.model, test_dataset)
        print('fid :', fid)
        fids.append(fid)

    Evaluate.save_graph(fids)


def test_gan():
    generator = Model.Generator()
    generator.load()
    _, test_dataset = Data.load_dataset()

    fid = Evaluate.get_fid(generator.model, test_dataset)
    print('fid :', fid)
