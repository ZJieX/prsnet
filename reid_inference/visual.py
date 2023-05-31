import os
import matplotlib.pyplot as plt
from PIL import Image


if __name__ == "__main__":
    path_infor = dict()

    im_show = False
    file_path = "result/msmt17/similer_imgs.txt"
    save_path = "result/msmt17/msmt17_visual/"

    if os.path.exists(save_path):
        os.makedirs(save_path)

    file =  open(file_path, "r", encoding="utf-8")
    for f in file:
        id = int(f.split(":")[0].split("/")[-1].split("_")[0])
        path_infor.setdefault(id, []).append(f)

    for p_id in path_infor:
        print(p_id)
        fig = plt.figure()
        row = len(path_infor[p_id])

        for k, pi in enumerate(path_infor[p_id]):
            q_path = pi.split(":")[0]
            id = int(q_path.split("/")[-1].split("_")[0])
            q_img = Image.open(q_path)
            # print("{}:{}".format(pi, q_path))
            Rank10 = pi.split(":")[1].split(",")
            col = len(Rank10) + 1

            ax = fig.add_subplot(row, col, 1+k*11, xticks=[], yticks=[])
            if 1 + k * 11 == 1:
                ax.set_title("{}".format("q_id:{}".format(id)), color=("black"), fontsize=8, ha='center')
            plt.imshow(q_img)

            for i, r in enumerate(Rank10):
                R_img_path = r.strip()
                R_img = Image.open(R_img_path)

                ax = fig.add_subplot(row, col, i + 2 + 11 * k, xticks=[], yticks=[])
                if i + 2 + 11 * k == 2 or i + 2 + 11 * k == 6 or i + 2 + 11 * k == 11:
                    ax.set_title("{}".format("Rank{}".format((i + 2) - 1)), color=("black"), fontsize=8, ha='center')
                plt.imshow(R_img)
        plt.savefig(save_path + "{}.png".format(int(p_id)))
        if im_show:
            plt.show()
