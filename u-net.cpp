#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dnn.h>


using namespace std;
using namespace dlib;

int train_images_num = 710;
int test_images_num = 20;
int train_bias = 0;
int test_bias = 0;

unsigned long int TP = 0;
unsigned long int TN = 0;
unsigned long int FP = 0;
unsigned long int FN = 0;

double dice_coefficient = 0;

int main(int argc, char** argv)
{   

    std::vector<matrix<rgb_pixel>> train_images;
    std::vector<matrix<uint16_t>> train_labels;
    std::vector<matrix<rgb_pixel>> test_images;
    std::vector<matrix<uint16_t>> test_labels;

    std::vector<matrix<unsigned char>> test_inputs;
    std::vector<matrix<unsigned char>> train_inputs;

    std::vector<matrix<unsigned char>> output_images;
    std::vector<matrix<uint16_t>> output_labels;

    std::vector<matrix<rgb_pixel>> test_images_resize;
    std::vector<matrix<unsigned char>> test_inputs_resize;

    std::vector<matrix<rgb_pixel>> train_images_resize;
    std::vector<matrix<unsigned char>> train_inputs_resize;

    train_images.clear();
    train_images.resize(train_images_num);
    for(int i=0;i<train_images_num;i++){train_images[i].set_size(512,512);}

    train_labels.clear();
    train_labels.resize(train_images_num);

    test_images.clear();
    test_images.resize(test_images_num);
    for(int i=0;i<test_images_num;i++){test_images[i].set_size(512,512);}

    test_labels.clear();
    test_labels.resize(test_images_num);

    test_inputs.clear();
    test_inputs.resize(test_images_num);
    for(int i=0;i<test_images_num;i++){test_inputs[i].set_size(512,512);}

    train_inputs.clear();
    train_inputs.resize(train_images_num);
    for(int i=0;i<train_images_num;i++){train_inputs[i].set_size(512,512);}

    output_images.clear();
    output_images.resize(test_images_num);

    output_labels.clear();
    output_labels.resize(test_images_num);

    test_images_resize.clear();
    test_images_resize.resize(test_images_num);

    test_inputs_resize.clear();
    test_inputs_resize.resize(test_images_num);

    train_images_resize.clear();
    train_images_resize.resize(train_images_num);

    train_inputs_resize.clear();
    train_inputs_resize.resize(train_images_num);

    stringstream str;

    for(int i=train_bias;i<train_bias+train_images_num;i++){
        str << "/home/salar/Desktop/nuclei_dataset3/" << i << "_image.png";
        load_image(train_images_resize[i-train_bias], str.str());
        resize_image(train_images_resize[i-train_bias], train_images[i-train_bias], interpolate_bilinear());
        str.str("");

        str << "/home/salar/Desktop/nuclei_dataset3/" << i << "_mask.png";
        load_image(train_inputs_resize[i-train_bias], str.str());
        resize_image(train_inputs_resize[i-train_bias], train_inputs[i-train_bias], interpolate_bilinear());
        str.str("");
        printf("%d.Train image is loading...\n", i-train_bias+1);
    }

    for(int i=test_bias;i<test_bias+test_images_num;i++){
        str << "/home/salar/Desktop/Dlib_U-net/nuclei_dataset1/" << i << "_image.png";
        load_image(test_images_resize[i-test_bias], str.str());
        resize_image(test_images_resize[i-test_bias], test_images[i-test_bias], interpolate_bilinear());
        str.str("");

        str << "/home/salar/Desktop/Dlib_U-net/nuclei_dataset1/" << i << "_mask.png";
        load_image(test_inputs_resize[i-test_bias], str.str());
        resize_image(test_inputs_resize[i-test_bias], test_inputs[i-test_bias], interpolate_bilinear());
        str.str("");
        printf("%d.Test image is loading...\n", i+1);
    }

    for(int k=0;k<train_images_num;k++){
        for(int j=0;j<512;j++){
            for(int i=0;i<512;i++){
                train_labels[k].set_size(512,512);
                if(train_inputs[k](j,i)>100) train_labels[k](j,i) = 1;
                else train_labels[k](j,i) = 0;
            }
        }
        printf("%d.Train label is being processed...\n", k+1);
    }

    for(int k=0;k<test_images_num;k++){
        for(int j=0;j<512;j++){
            for(int i=0;i<512;i++){
                test_labels[k].set_size(512,512);
                if(test_inputs[k](j,i)>100) test_labels[k](j,i) = 1;
                else test_labels[k](j,i) = 0;
            }
        }
        printf("%d.Test label is being processed...\n", k+1);
    }

    using net_type = loss_multiclass_log_per_pixel<                         //Output
                                relu<con<2,3,3,1,1,                         //Transposed Convolution(1)
                                dropout<relu<con<16,3,3,1,1,                //Convolution(16)
                                concat2<tag9,tag10,                         //Concatenation
                                tag10<cont<16,6,6,2,2,relu<con<32,3,3,1,1,  //Transposed Convolution(16)
                                dropout<relu<con<32,3,3,1,1,                //Convolution(32)
                                concat2<tag7,tag8,                          //Concatenation
                                tag8<cont<32,6,6,2,2,relu<con<64,3,3,1,1,   //Transposed Convolution(32)
                                dropout<relu<con<64,3,3,1,1,                //Convolution(64)
                                concat2<tag5,tag6,                          //Concatenation
                                tag6<cont<64,5,5,2,2,relu<con<128,3,3,1,1,  //Transposed Convolution(64)
                                dropout<relu<con<128,3,3,1,1,               //Convolution(128)
                                concat2<tag3,tag4,                          //Concatenation
                                tag4<cont<128,5,5,2,2,relu<con<256,3,3,1,1, //Transposed Convolution(128)
                                dropout<relu<con<256,3,3,1,1,               //Convolution(256)
                                concat2<tag1,tag2,                          //Concatenation
                                tag2<cont<256,5,5,2,2,relu<con<512,3,3,1,1, //Transposed Convolution(256)
                                dropout<relu<con<512,3,3,1,1,               //Convolution(512)
                                max_pool<5,5,2,2,tag1<relu<con<256,3,3,1,1, //Max Pooling
                                dropout<relu<con<256,3,3,1,1,               //Convolution(256)
                                max_pool<5,5,2,2,tag3<relu<con<128,3,3,1,1, //Max Pooling
                                dropout<relu<con<128,3,3,1,1,               //Convolution(128)
                                max_pool<5,5,2,2,tag5<relu<con<64,3,3,1,1,  //Max Pooling
                                dropout<relu<con<64,3,3,1,1,                //Convolution(64)
                                max_pool<5,5,2,2,tag7<relu<con<32,3,3,1,1,  //Max Pooling
                                dropout<relu<con<32,3,3,1,1,                //Convolution(32)
                                max_pool<5,5,2,2,tag9<relu<con<16,3,3,1,1,  //Max Pooling
                                dropout<relu<con<16,3,3,1,1,                //Convolution(16)
                                input<matrix<rgb_pixel>>                    //Input
                                >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

    net_type net;

    dnn_trainer<net_type> trainer(net);

    trainer.set_learning_rate(0.01);
    trainer.set_min_learning_rate(0.00001);
    trainer.set_mini_batch_size(2);
    trainer.set_max_num_epochs(25);
    trainer.be_verbose();

    trainer.set_synchronization_file("u-net_sync", std::chrono::seconds(20));

    trainer.train(train_images, train_labels);

    net.clean();
    serialize("u-net_network.dat") << net;

    for(int i=0;i<test_images_num;i++){
        output_labels[i] = net(test_images[i]);
    }

    for(int k=0;k<test_images_num;k++){
        for(int j=0;j<512;j++){
            for(int i=0;i<512;i++){
                output_images[k].set_size(512,512);
                if(output_labels[k](j,i)>0) output_images[k](j,i) = 255;
                else output_images[k](j,i) = 0;
            }
        }        
    }

    matrix<rgb_pixel> empty_image;
    image_window my_window1(empty_image, "Predicted Mask");
    image_window my_window2(empty_image, "Processed Mask");
    image_window my_window3(empty_image, "Original Image");

    for(int p=0;p<test_images_num;p++){
        my_window1.set_image(output_images[p]);
        my_window2.set_image(test_inputs[p]);
        my_window3.set_image(test_images[p]);

        for(int j=0;j<512;j++){
            for(int i=0;i<512;i++){
                if(test_inputs[p](j,i)==0){
                    if(output_images[p](j,i)==0) TN++;
                    else FP++;
                }
                else{
                    if(output_images[p](j,i)==0) FN++;
                    else TP++;
                }
            }
        }

        dice_coefficient = (2.0*TP)/(double)(FN+(2*TP)+FP);
        printf("Dice Coefficient: %lf\n", dice_coefficient);

        TP = 0;
        TN = 0;
        FP = 0;
        FN = 0;

        if(p<test_images_num-1){
            printf("Press enter to get next prediciton...\n");
            getchar();
        }
        else{
            printf("Press enter to quit!\n");
            getchar();
        }
    }
    
    return 0;

}