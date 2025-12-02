# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from reconstruction.helpers.homography import ransac_homography, warp_and_stitch

def run():
    A = cv2.imread("reconstruction/data/a.jpg")
    B = cv2.imread("reconstruction/data/b.jpg")
    assert A is not None and B is not None, "Missing a.jpg or b.jpg"

    # SIFT (VLFeat equivalent)
    sift = cv2.SIFT_create()
    kpA, dA = sift.detectAndCompute(cv2.cvtColor(A, cv2.COLOR_BGR2GRAY), None)
    kpB, dB = sift.detectAndCompute(cv2.cvtColor(B, cv2.COLOR_BGR2GRAY), None)

    bf = cv2.BFMatcher()
    matches = sorted(bf.match(dA, dB), key=lambda m: m.distance)

    xA = np.array([kpA[m.queryIdx].pt for m in matches]).T     # 2xN
    xB = np.array([kpB[m.trainIdx].pt for m in matches]).T

    # Visualize 10 random matches
    perm = np.random.permutation(len(matches))[:10]
    concat = np.concatenate([A, B], axis=1)
    plt.figure(); plt.imshow(cv2.cvtColor(concat, cv2.COLOR_BGR2RGB)); plt.axis('off')
    for i in perm:
        p1 = np.array(kpA[matches[i].queryIdx].pt)
        p2 = np.array(kpB[matches[i].trainIdx].pt) + np.array([A.shape[1], 0])
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-')
    plt.title("Random subset of matches")

    print(f"Number of SIFT features in Image A: {len(kpA)}")
    print(f"Number of SIFT features in Image B: {len(kpB)}")
    print(f"Number of matches: {len(matches)}")

    H, inliers = ransac_homography(xA, xB, num_iter=1000, thresh=5.0)
    print(f"Number of inliers: {int(inliers.sum())}")

    panorama = warp_and_stitch(A, B, H)
    plt.figure(); plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.title("Stitched panorama"); plt.axis('off')
    plt.show()

if __name__ == "__main__":
    run()

