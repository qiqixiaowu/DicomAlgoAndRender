#pragma once

#include <string>
#include <vector>
#include <optional>
#include <dcmtk/dcmdata/dctk.h>

struct SliceMeta {
    std::string filepath;
    int instanceNumber = -1;
    double imagePosition[3] = {0,0,0};
    double orientationRow[3] = {1,0,0};
    double orientationCol[3] = {0,1,0};
    bool hasPosition = false;
    bool hasOrientation = false;
};

struct SeriesData {
    std::string seriesUID;
    std::vector<SliceMeta> slices;
};

std::optional<std::string> readStringTag(DcmDataset* ds, uint16_t g, uint16_t e);
std::optional<int> readIntTag(DcmDataset* ds, uint16_t g, uint16_t e);
std::optional<std::vector<double>> readMultiNumber(DcmDataset* ds, uint16_t g, uint16_t e);
SeriesData collectSeries(const std::string &folder, const std::string &targetSeriesUID = "");
