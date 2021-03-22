# Cycle Self-Training for Instance Search in Real World

This code has the source code for the paper "Cycle Self-Training (CST) for Instance Search in Real World Applications". Including:

> the annotation of INS-PRW
> 
> the annotation of INS-CUHK-SYSU
> 
> CST code

================================================================

## About Annotation
* Please download original images from [CUHK-SYSU](https://github.com/ShuangLI59/person_search) and [PRW](http://zheng-lab.cecs.anu.edu.au/Project/project_prw.html). Other information including the naming style, image size, etc., can be also found in the original project pages.

* We provide txt files for annotation in folder `annotation`. The annotation protocal is similar to PRW, `gallery.txt` and `query.txt` inlucde annotation with the format as [ID, left, top, right, bottom, image_name]. Only objects that pass through at least 2 cameras are taken into account.

* In total, we have 16,780 bboxes for `INS_CUHK_SYSU_gallery.txt` and 6,972 bboxes for `INS-CUHK_SYSU_query.txt`. `INS_CUHK_SYSU_local_gallery.txt` is the local distractor for each query, which is similar to the local gallery setting in original CUHK_SYSU. We have 7,834 bboxes for `INS_PRW_gallery.txt` and 1,537 bboxes for `INS_PRW_query.txt`.

================================================================

## About CST
pending...
