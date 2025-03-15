# def property
image_folder = "/content/drive/MyDrive/Ai_Lab/data/coco/val2017"
ann_file = '/content/drive/MyDrive/Ai_Lab/data/coco/annotations_small/instances_val2017.json'
main_labels = {1, 3, 62, 44,17,84,18,16,70}
size = 224
num_classes = 183



##Run Function
#Y
filtered_dict,sorted_keys_list = one_label_COCO(image_folder, ann_file, main_labels)
my_dict = filtered_dict
y_all=onehot_labels(my_dict, num_classes)


#X
x_all = pic_preproc_Intersection_3chan(image_folder, ann_file,main_labels, size)


x_al=x_all
y_al = y_all
test_ratio = 0.2

x_train, x_test, y_train, y_test = train_test_splitt(x_al, y_al, test_ratio)

min_pixel_value = 0
max_pixel_value = 1
print ("min_pixel_value =" , min_pixel_value, "," , "max_pixel_value =" , max_pixel_value)
print ("x_train.shape =" , x_train.shape, "," , "y_train =" , y_train.shape)