#include <mvnc.h>
#include <vector>
#include <stdio.h>
#include <string>
#include <chrono>
#include <unistd.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "movidiusdevice.h"

const int req_width = 227;
const int req_height = 227;
const bool show_perfs = false;
const bool show_results = false;

int runNetwork(const std::vector<std::string>& fnames,
               const std::vector<unsigned char*>& images,
               movidius_device* movidius_dev,
               const std::string& networkPath)
{
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    std::chrono::high_resolution_clock::time_point t3;

    strcpy(movidius_dev->networkPath,  networkPath.c_str());
    int ret = movidius_uploadNetwork(movidius_dev);

    if (ret != 0)
    {
        fprintf(stderr, "Failed allocating graph: %d\n", ret);
        return 1;
    }

    float* results = NULL;
    if (movidius_dev->numCategories == 0)
    {
        fprintf(stderr, "no categories after loading network\n");
        return 1;
    }

    results = new float[movidius_dev->numCategories];
    memset(results, 0, sizeof(float) * movidius_dev->numCategories);

    for (int c = 0; c < fnames.size(); c++)
    {
        t1 = std::chrono::high_resolution_clock::now();
        if (movidius_convertImage((movidius_RGB*)images.at(c), req_width,req_height, movidius_dev) != 0)
        {
            fprintf(stderr, "failed converting image to 16bit float format\n");
            return 1;
        }

        t2 = std::chrono::high_resolution_clock::now();

        int ret = movidius_runInference(movidius_dev, results);

        t3 = std::chrono::high_resolution_clock::now();

        if (ret != 0)
        {
            fprintf(stderr, "runinference failure: %d for image %s\n", ret, fnames.at(c).c_str());
            return 1;
        }

        auto dur1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        auto dur2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

        if (show_perfs)
        {
            fprintf(stderr, "convertImage(): %d us\n", dur1);
            fprintf(stderr, "runInference(): %d us\n", dur2);
        }

        if (show_results)
        {
            for (int cat = 0; cat < movidius_dev->numCategories; cat++)
            {
                fprintf(stderr, "category %d (%s): %f\n", cat, movidius_dev->categories[cat], results[cat]);
            }
        }
    }

    ret = movidius_deallocateGraph(movidius_dev);
    if (ret != 0)
    {
        fprintf(stderr, "Failed deallocating graph: %d\n", ret);
        return 1;
    }

    delete[] results;
    results = NULL;
    return 0;
}

int main(int argc, char** argv)
{
    movidius_device movidius_dev;
    memset(&movidius_dev, 0, sizeof(movidius_device));

    if (movidius_openDevice(&movidius_dev) != 0)
        return 1;

    std::vector<unsigned char*> images;
    std::vector<std::string> fnames;

    fnames.push_back("./sample_1505732941144.png");
    fnames.push_back("./sample_1505732942167.png");
    fnames.push_back("./sample_1505732945220.png");
    fnames.push_back("./sample_1505732946240.png");
    fnames.push_back("./sample_1505732947259.png");
    fnames.push_back("./sample_1505732948277.png");
    fnames.push_back("./sample_1505732949297.png");
    fnames.push_back("./sample_1505732952353.png");

    // load images from disk
    for (int c = 0; c < fnames.size(); c++)
    {
        int width, height, cp;
        width = height = cp = 0;

        unsigned char* img = stbi_load(fnames.at(c).c_str(), &width, &height, &cp, 3); 
        if (img == NULL)
        {
            fprintf(stderr, "The image %s could not be loaded\n", fnames.at(c).c_str());
            return 1;
        }

        if (width != req_width || height != req_height)
        {
            fprintf(stderr, "Invalid size for image %s. Expected %d x %d, got %d x %d\n",
                    fnames.at(c).c_str(), width, height, req_width, req_height);
            return 1;
        }

        images.push_back(img);
    }

    int loops = 0;
    int loops_total = 4;
    int ret = 0;

    // run networks a few times
    while (loops < loops_total)
    {
        loops++;

        ret = runNetwork(fnames, images, &movidius_dev, "./network/Age");
        if (ret != 0)
        {
            fprintf(stderr, "Age network failed: %d\n", ret);
            break;
        }

        ret = runNetwork(fnames, images, &movidius_dev, "./network/Gender");
        if (ret != 0)
        {
            fprintf(stderr, "Gender network failed: %d\n", ret);
            break;
        }

        ret = runNetwork(fnames, images, &movidius_dev, "./network/Age");
        if (ret != 0)
        {
            fprintf(stderr, "Age network failed: %d\n", ret);
            break;
        }

        ret = runNetwork(fnames, images, &movidius_dev, "./network/Gender");
        if (ret != 0)
        {
            fprintf(stderr, "Gender network failed: %d\n", ret);
            break;
        }

        ret = runNetwork(fnames, images, &movidius_dev, "./network/Age");
        if (ret != 0)
        {
            fprintf(stderr, "Age network failed: %d\n", ret);
            break;
        }

        ret = runNetwork(fnames, images, &movidius_dev, "./network/Gender");
        if (ret != 0)
        {
            fprintf(stderr, "Gender network failed: %d\n", ret);
            break;
        }

        fprintf(stderr, "Done with loop %d / %d\n", loops, loops_total);
    }

    for (int c = 0; c < images.size(); c++)
        free(images.at(c));
    images.clear();

    movidius_closeDevice(&movidius_dev, false);

    return ret;
}
