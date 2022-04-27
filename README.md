

# FYP-Unsupervised Domain Adaptation for bearing fault diagnosis

## XJTU Dataset Selection

 - Only choose stable data.  Data with significant shakes will be dropped.

| Dataset          | Element | Good (Normal) Starts at | Fault Starts at |
| ---------------- | ------- | ----------------------- | --------------- |
| Bearing 2_1 37.5 | Inner   | 1-452                   | 454-484         |
| Bearing 2_2      | Outer   | 1-50                    | 51-159          |
| Bearing 2_4      | Outer   | 1-30                    | 31-40           |
| Bearing 2_5      | Outer   | 1-120                   | 121-337         |
| Bearing 3_1 40Hz | Outer   | 1-2463                  | 2464-2536       |
| Bearing 3_3      | Inner   | 1-340                   | 341-369         |
| Bearing 3_4      | Inner   | 1-1416                  | 1417-1514       |
| Bearing 3_5      | Outer   | 1-10                    | 11-110          |

## Methodology

Purposed two different types of Unsupervised Domain Adaptation in adversarial manners, and applied pseudo label semi-supervised learning strategy. The two different types of models are compared and analyzed. 

### Proposed structure of feature extractor



![image-20220427173456789](readme-image\image-20220427173456789.png)

### First type of Unsupervised Domain Adaptation model

![image-20220427173803408](readme-image\image-20220427173803408.png)



## Experiment results

### Confusion matrix of the first proposed model

![image-20220427174004039](readme-image\image-20220427174004039.png)



### Loss and accuracy of the second proposed model

![image-20220427174035953](readme-image\image-20220427174035953.png)

