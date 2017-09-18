#include "movidiusdevice.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "movidius_fp16.h"

void printMovidiusError(int rc)
{
    switch (rc)
    {
    case MVNC_OK: fprintf(stderr, "MVNC_OK\n"); break;
    case MVNC_BUSY: fprintf(stderr, "MVNC_BUSY\n"); break;
    case MVNC_ERROR: fprintf(stderr, "MVNC_ERROR\n"); break;
    case MVNC_OUT_OF_MEMORY: fprintf(stderr, "MVNC_OUT_OF_MEMORY\n"); break;
    case MVNC_DEVICE_NOT_FOUND: fprintf(stderr, "MVNC_DEVICE_NOT_FOUND\n"); break;
    case MVNC_INVALID_PARAMETERS: fprintf(stderr, "MVNC_INVALID_PARAMETERS\n"); break;
    case MVNC_TIMEOUT: fprintf(stderr, "MVNC_TIMEOUT\n"); break;
    case MVNC_MVCMDNOTFOUND: fprintf(stderr, "MVNC_MVCMDNOTFOUND\n"); break;
    case MVNC_NODATA: fprintf(stderr, "MVNC_NODATA\n"); break;
    case MVNC_GONE: fprintf(stderr, "MVNC_GONE\n"); break;
    case MVNC_UNSUPPORTEDGRAPHFILE: fprintf(stderr, "MVNC_UNSUPPORTEDGRAPHFILE\n"); break;
    case MVNC_MYRIADERROR: fprintf(stderr, "MVNC_MYRIADERROR\n"); break;
    default: fprintf(stderr, "Unknown error code %d\n", rc); break;
    }
}

void movidius_convertImage(RGB* colorimage, unsigned int color_width, unsigned int color_height, movidius_device* dev)
{
    if (color_width != dev->reqsize || color_height != dev->reqsize)
    {
        fprintf(stderr, "movidius: error, given image is wrong size: %d, %d. Expecting size: %d, %d\n", color_width, color_height, dev->reqsize, dev->reqsize);
        return;
    }

    if (dev->currentImageSize != dev->reqsize)
    {
        if (dev->movidius_image != NULL)
            free(dev->movidius_image);
        dev->movidius_image = (movidius_RGB_f16*)malloc(sizeof(movidius_RGB_f16) * dev->reqsize * dev->reqsize);
        memset(dev->movidius_image, 0, sizeof(uint16_t) * dev->reqsize * dev->reqsize * 3);

        if (dev->scaled_image != NULL)
            free(dev->scaled_image);

        dev->scaled_image = (float*)malloc(sizeof(float) * dev->reqsize * dev->reqsize * 3);
        memset(dev->scaled_image, 0, sizeof(float) * dev->reqsize * dev->reqsize * 3);
    }

    RGB temp;
    for (int y = 0; y < dev->reqsize; y++)
    {
        for (int x = 0; x < dev->reqsize; x++)
        {
            temp = colorimage[y * color_width + x];
            dev->scaled_image[3 * (y * dev->reqsize + x) + 0] = (((float)temp.r) - dev->mean[0]) * dev->standard_deviation[0];
            dev->scaled_image[3 * (y * dev->reqsize + x) + 1] = (((float)temp.g) - dev->mean[1]) * dev->standard_deviation[1];
            dev->scaled_image[3 * (y * dev->reqsize + x) + 2] = (((float)temp.b) - dev->mean[2]) * dev->standard_deviation[2];
        }
    }

    floattofp16((unsigned char*)dev->movidius_image, dev->scaled_image, 3*dev->reqsize*dev->reqsize);
}

int movidius_runInference(movidius_device* dev)
{
    for (int c = 0; c < dev->inferenceCount; c++)
    {
        int i, throttling;
        void* result;
        unsigned int resultlen;
        void* userParam;
        float* cmpdata;
        float* timetaken;
        unsigned int timetakenlen, throttlinglen;
        float* resultfp32;
        int* indexes;

        int rc = mvncLoadTensor(dev->currentGraphHandle, dev->movidius_image, 3 * dev->reqsize * dev->reqsize * sizeof(*dev->movidius_image), 0);
        if (rc != MVNC_OK)
        {
            fprintf(stderr, "movidius: LoadTensor failed: %d\n", rc);
            printMovidiusError(rc);
            return MOVIDIUS_LOADTENSOR_ERROR;
        }

        rc = mvncGetResult(dev->currentGraphHandle, &result, &resultlen, &userParam);
        if (rc != MVNC_OK)
        {
            if (rc == MVNC_MYRIADERROR)
            {
                char* debuginfo;
                unsigned debuginfolen;

                rc = mvncGetGraphOption(dev->currentGraphHandle, MVNC_DEBUGINFO, (void**)&debuginfo, &debuginfolen);
                if (rc == MVNC_OK)
                {
                    fprintf(stderr, "movidius: GetResult failed, myriad error: %s\n", debuginfo);
                    return 1;
                }
            }

            fprintf(stderr, "movidius: GetResult failed, rc=%d\n", rc);
            printMovidiusError(rc);
            return 1;
        }

        resultlen /= sizeof(unsigned short);
        resultfp32 = (float*)malloc(resultlen * sizeof(*resultfp32));
        fp16tofloat(resultfp32, (unsigned char*)result, resultlen);

        indexes = (int*)malloc(sizeof(*indexes) * resultlen);
        for(i = 0; i < (int)resultlen; i++)
            indexes[i] = i;
        cmpdata = resultfp32;

        movidius_sortInference(indexes, indexes + resultlen, cmpdata);

        for(i = 0; i < 5 && i < resultlen; i++)
        {
            fprintf(stderr, "movidius: %s (%.2f%%) ", dev->categories[indexes[i]], resultfp32[indexes[i]] * 100.0);
        }

        fprintf(stderr, "\n");
        free(resultfp32);
        free(indexes);

        rc = mvncGetGraphOption(dev->currentGraphHandle, MVNC_TIMETAKEN, (void **)&timetaken, &timetakenlen);
        if(rc)
        {
            fprintf(stderr, "movidius: GetGraphOption failed for getting MVNC_TIMETAKEN, rc=%d\n", rc);
            printMovidiusError(rc);
            return 1;
        }

        timetakenlen = timetakenlen / sizeof(*timetaken);
        float sum = 0;
        for(i = 0; i < timetakenlen; i++)
            sum += timetaken[i];

        fprintf(stderr, "movidius: Inference time: %f ms\n", sum);

        rc = mvncGetDeviceOption(dev->dev_handle, MVNC_THERMAL_THROTTLING_LEVEL, (void **)&throttling, &throttlinglen);
        if (rc)
        {
            fprintf(stderr, "movidus: GetGraphOption failed for MVNC_THERMAL_THROTTLING_LEVEL, rc=%d\n", rc);
            printMovidiusError(rc);
            return 1;
        }

        if (throttling == 1)
            fprintf(stderr, "movidius: ** NCS temperature high - thermal throttling initiated **\n");
        else if (throttling == 2)
        {
            fprintf(stderr, "movidius: *********************** WARNING *************************\n");
            fprintf(stderr, "movidius: * NCS temperature critical                              *\n");
            fprintf(stderr, "movidius: * Aggressive thermal throttling initiated               *\n");
            fprintf(stderr, "movidius: * Continued use may result in device damage             *\n");
            fprintf(stderr, "movidius: *********************************************************\n");
        }
    }

    return 0;
}

void* movidius_loadfile(const char* path, unsigned int* length)
{
    FILE *fp;
    char *buf;

    fp = fopen(path, "rb");
    if (fp == NULL)
    {
        fprintf(stderr, "movidius: Cannot read file: %s\n", path);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    *length = ftell(fp);
    rewind(fp);

    if (!(buf = (char*)malloc(*length)))
    {
        fclose(fp);
        fprintf(stderr, "movidius: Cannot allocate buffer of size %d for file %s\n", *length, path);
        return NULL;
    }

    if (fread(buf, 1, *length, fp) != *length)
    {
        fclose(fp);
        free(buf);
        fprintf(stderr, "movidius: Failed reading %d bytes from file %s\n", *length, path);
        return NULL;
    }

    fclose(fp);
    return buf;
}

int movidius_loadGraphData(const char* dir, unsigned int* reqsize, float* mean, float* std)
{
    char path[300];
    int i;

    snprintf(path, sizeof(path), "%s/stat.txt", dir);
    FILE *fp = fopen(path, "r");
    if (!fp)
    {
        fprintf(stderr, "movidius: Failed opening stat.txt in dir: %s\n", dir);
        return -1;
    }

    if (fscanf(fp, "%f %f %f\n%f %f %f\n", mean, mean+1, mean+2, std, std+1, std+2) != 6)
    {
        fprintf(stderr, "movidius: %s: mean and stddev not found in file\n", path);
        fclose(fp);
        return -1;
    }

    fclose(fp);

    for (i = 0; i < 3; i++)
    {
        mean[i] = 255.0 * mean[i];
        std[i] = 1.0 / (255.0 * std[i]);
    }

    snprintf(path, sizeof(path), "%s/inputsize.txt", dir);
    fp = fopen(path, "r");
    if (!fp)
    {
        fprintf(stderr, "movidius: Failed opening inputsize.txt in dir %s\n", dir);
        return -1;
    }

    if (fscanf(fp, "%d", reqsize) != 1)
    {
        fprintf(stderr, "movidius: %s: inputsize not found in file\n", path);
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

int movidius_loadCategories(const char* path, movidius_device* dev)
{
    char line[300], *p;
    FILE *fp = fopen(path, "r");

    if (!fp)
    {
        fprintf(stderr, "movidius: Failed opening file: %s\n", path);
        return -1;
    }

    dev->numCategories = 0;
    dev->categories = (char**)malloc(1000 * sizeof(*dev->categories));

    while (fgets(line, sizeof(line), fp))
    {
        p = strchr(line, '\n');
        if (p)
            *p = 0;

        if (strcasecmp(line, "classes"))
        {
            dev->categories[dev->numCategories++] = strdup(line);
            if (dev->numCategories == 1000)
                break;
        }
    }

    fclose(fp);
    return 0;
}

int movidius_uploadNetwork(movidius_device* dev)
{
    if (dev->dev_handle == NULL)
    {
        fprintf(stderr, "movidius: cannot load graph for null device\n");
        return -1;
    }

    if (strlen(dev->networkPath) == 0)
    {
        fprintf(stderr, "movidius: No network file path given\n");
        return -1;
    }

    if (dev->graphFileContents != NULL || dev->currentGraphHandle != NULL || dev->numCategories > 0 || dev->categories != NULL)
    {
        fprintf(stderr, "movidius: Cannot upload a new network before calling movidius_deallocateGraph\n");
        return 1;
    }

    int rc, i;
    void* g = NULL;
    char path[1024];
    unsigned len;

    snprintf(path, sizeof(path), "%s/graph", dev->networkPath);
    dev->graphFileContents = movidius_loadfile(path, &len);

    if (dev->graphFileContents == NULL)
    {
        fprintf(stderr, "movidius: %s/graph not found\n", dev->networkPath);
        return 1;
    }

    snprintf(path, sizeof(path), "%s/categories.txt", dev->networkPath);

    if (movidius_loadCategories(path, dev) != 0)
    {
        fprintf(stderr, "movidius: Error loading categories\n");
        free(dev->graphFileContents);
        dev->graphFileContents = NULL;
        return 1;
    }

    // TODO: what is this?
    dev->inferenceCount = 1;

    if (!movidius_loadGraphData(dev->networkPath, &dev->reqsize, dev->mean, dev->standard_deviation))
    {
        rc = mvncAllocateGraph(dev->dev_handle, &g, dev->graphFileContents, len);

        if (rc != MVNC_OK)
        {
            fprintf(stderr, "movidius: AllocateGraph failed, rc = %d\n", rc);
            printMovidiusError(rc);
            free(dev->graphFileContents);
            for (i = 0; i < dev->numCategories; i++)
                free(dev->categories[i]);
            free(dev->categories);
            dev->graphFileContents = NULL;
            dev->categories = NULL;
            dev->numCategories = 0;

            return 1;
        }

        dev->currentGraphHandle = g;

        fprintf(stderr, "movidius: Graph allocated\n");
    }

    return 0;
}

int movidius_deallocateGraph(movidius_device* dev)
{
    if (dev->dev_handle == NULL)
    {
        fprintf(stderr, "movidius: cannot unload graph for null device\n");
        return -1;
    }

    if (dev->currentGraphHandle == NULL)
    {
        fprintf(stderr, "movidius: cannot unload null graph\n");
        return -1;
    }

    if (dev->numCategories > 0)
    {
        for (int i = 0; i < dev->numCategories; i++)
            free(dev->categories[i]);
        free(dev->categories);
    }

    dev->categories = NULL;
    dev->numCategories = 0;
    if (dev->graphFileContents != NULL)
        free(dev->graphFileContents);
    dev->graphFileContents = NULL;

    // Apparently this function may not be implemented in all versions?
    // If so, how do we free memory?
    int rc = mvncDeallocateGraph(dev->currentGraphHandle);

    if (rc != MVNC_OK)
    {
        fprintf(stderr, "movidius: Failed deallocating graph: %d\n", rc);
        printMovidiusError(rc);

        // Assume these errors happen if the device does not support deallocating graphs
        // I suppose we have to restart device then? Well, assign this pointer to NULL
        // as we've done all we can
        dev->currentGraphHandle = NULL;
        return -1;
    }

    dev->currentGraphHandle = NULL;

    return 0;
}

int movidius_openDevice(movidius_device* dev)
{
    assert(sizeof(dev->dev_name) >= MVNC_MAXNAMESIZE);

    char name[MVNC_MAXNAMESIZE];
    int loglevel = 2;
    void* h = NULL;

    mvncSetDeviceOption(0, MVNC_LOGLEVEL, &loglevel, sizeof(loglevel));

    int rc = mvncGetDeviceName(0, name, sizeof(name));
    if (rc != MVNC_OK)
    {
        fprintf(stderr, "movidius: No devices found\n");
        printMovidiusError(rc);
        return -1;
    }

    rc = mvncOpenDevice(name, &h);
    if (rc != MVNC_OK)
    {
        fprintf(stderr, "movidius: OpenDevice %s failed, rc=%d\n", name, rc);
        printMovidiusError(rc);
        return -1;
    }

    fprintf(stderr, "movidius: OpenDevice %s succeeded\n", name);
    dev->dev_handle = h;
    strcpy(dev->dev_name, name);

    return 0;
}

int movidius_closeDevice(movidius_device* dev)
{
    if (dev->dev_handle == NULL)
    {
        fprintf(stderr, "movidius: cannot close null device\n");
        return 1;
    }

    if (dev->scaled_image)
        free(dev->scaled_image);
    dev->scaled_image = NULL;

    if (dev->movidius_image)
        free(dev->movidius_image);
    dev->movidius_image = NULL;

    dev->currentImageSize = 0;

    movidius_deallocateGraph(dev);

    int rc = mvncCloseDevice(dev->dev_handle);
    if (rc != MVNC_OK)
    {
        fprintf(stderr, "movidius: Device close failed: %d\n", rc);
        printMovidiusError(rc);
        return 1;
    }

    fprintf(stderr, "movidius: Device closed\n");
    return 0;
}

#ifdef __cplusplus
}
#endif
