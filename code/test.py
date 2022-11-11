from model_base import *
from model_mixture_of_exprts import *
from model_product_of_exprts import *
from helpers import *
from collections import deque
import imageio


def generate_results(test_x,
                     test_y,
                     model,
                     model_config,
                     save_fig=0,
                     save_conf_fig=0,
                     save_calibration=0):

    # to do add some calibration plots

    test_x_o = np.shape(test_x)[0]

    predictions = deque()
    probabilities = deque()
    error_maps = deque()

    for i in range(test_x_o):
        data = np.expand_dims(test_x[i], axis=0)
        data = torch.from_numpy(data).type(torch.FloatTensor).to('cuda')
        prob = model(data)
        pred = (prob > 0.5).float()
        predictions.append(pred)
        probabilities.append(prob)

        # get errors:
        h, w = np.shape(pred)
        error = np.zeros((h, w, 3))
        lbl = test_y[i]
        error[(pred == 1) & (lbl == 0)] = [255, 0, 0]  # false positives
        error[(pred == 0) & (lbl == 1)] = [0, 255, 0]  # false negatives
        error[(pred == 1) & (lbl == 1)] = [0, 0, 255]  # true positives
        error_maps.append(error)

        # save the segmentation:
        if save_fig == 1:
            imageio.imsave('../results/seg' + str(i) + '.png', pred)

        # save the error
        if save_conf_fig == 1:
            imageio.imsave('../results/error' + str(i) + '.png', error)

    np.save('../results/' + model_config + '_preds.npy', predictions)
    np.save('../results/' + model_config + '_probs.npy', probabilities)
    np.save('../results/' + model_config + '_errors.npy', error_maps)

    # # def display(rows, columns, images, values=[], predictions=[]):
    # fig = plt.figure(figsize=(9, 11))
    # ax = []
    #
    # for i in range(9):
    #     img, lbl, pred, error = test_x[i], test_y[i], predictions[i], error_maps[i]
    #     ax.append(fig.add_subplot(9, 4, i + 1))
    #
    #     title = ""
    #
    #     if (len(values) == 0):
    #         title = "Pred:" + str(predictions[i])
    #     elif (len(predictions) == 0):
    #         title = "Value:" + str(values[i])
    #     elif (len(values) != 0 and len(predictions) != 0):
    #         title = "Value:" + str(values[i]) + "\nPred:" + str(predictions[i])
    #
    #     ax[-1].set_title(title)  # set title
    #     plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    #
    # plt.show()

    return predictions


