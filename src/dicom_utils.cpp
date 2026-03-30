#pragma once

#include <glad/glad.h>

#include "dicom_utils.hpp"
#include <filesystem>
#include <algorithm>
#include <cmath>

std::optional<std::string> readStringTag(DcmDataset* ds, uint16_t g, uint16_t e) {
    if(!ds) return std::nullopt;
    DcmTagKey key(g,e);
    OFString value;
    if(ds->findAndGetOFString(key, value).good()){
        std::string v = value.c_str();
        while(!v.empty() && (v.back()==' ' || v.back()=='\0' || v.back()=='\r' || v.back()=='\n')) v.pop_back();
        return v;
    }
    return std::nullopt;
}

std::optional<int> readIntTag(DcmDataset* ds, uint16_t g, uint16_t e) {
    auto s = readStringTag(ds,g,e);
    if(!s) return std::nullopt;
    try { return std::stoi(*s); } catch(...) { return std::nullopt; }
}

std::optional<std::vector<double>> readMultiNumber(DcmDataset* ds, uint16_t g, uint16_t e) {
    auto s = readStringTag(ds,g,e);
    if(!s) return std::nullopt;
    std::vector<double> out;
    size_t start=0;
    while(start < s->size()) {
        size_t comma = s->find('\\', start);
        std::string token = s->substr(start, comma==std::string::npos? std::string::npos : comma-start);
        try { out.push_back(std::stod(token)); } catch(...) { out.push_back(0.0); }
        if(comma==std::string::npos) break;
        start = comma+1;
    }
    return out;
}

SeriesData collectSeries(const std::string &folder, const std::string &targetSeriesUID) {
    SeriesData result;
    std::string chosenSeries = targetSeriesUID;
    for(auto &entry : std::filesystem::directory_iterator(folder)) {
        if(!entry.is_regular_file()) continue;
        auto path = entry.path().string();

        DcmFileFormat fileformat;
        if(!fileformat.loadFile(path.c_str()).good()) continue;
        DcmDataset* ds = fileformat.getDataset();
        if(!ds) continue;

        auto seriesUIDOpt = readStringTag(ds, 0x0020, 0x000E); // SeriesInstanceUID
        if(!seriesUIDOpt) continue;
        const std::string &seriesUID = *seriesUIDOpt;

        if(chosenSeries.empty()) chosenSeries = seriesUID;
        if(seriesUID != chosenSeries) continue;

        SliceMeta meta;
        meta.filepath = path;
        meta.instanceNumber = readIntTag(ds, 0x0020, 0x0013).value_or(-1);

        if(auto pos = readMultiNumber(ds, 0x0020, 0x0032); pos && pos->size()>=3) {
            meta.imagePosition[0]=(*pos)[0];
            meta.imagePosition[1]=(*pos)[1];
            meta.imagePosition[2]=(*pos)[2];
            meta.hasPosition = true;
        }
        if(auto ori = readMultiNumber(ds, 0x0020, 0x0037); ori && ori->size()>=6) {
            for(int i=0;i<3;++i) meta.orientationRow[i]=(*ori)[i];
            for(int i=0;i<3;++i) meta.orientationCol[i]=(*ori)[i+3];
            meta.hasOrientation = true;
        }
        result.slices.push_back(meta);
    }
    result.seriesUID = chosenSeries;

    double N[3] = {0,0,1};
    if(!result.slices.empty() && result.slices.front().hasOrientation) {
        auto &m = result.slices.front();
        N[0] = m.orientationRow[1]*m.orientationCol[2] - m.orientationRow[2]*m.orientationCol[1];
        N[1] = m.orientationRow[2]*m.orientationCol[0] - m.orientationRow[0]*m.orientationCol[2];
        N[2] = m.orientationRow[0]*m.orientationCol[1] - m.orientationRow[1]*m.orientationCol[0];
        double len = std::sqrt(N[0]*N[0]+N[1]*N[1]+N[2]*N[2]);
        if(len>1e-6){ N[0]/=len; N[1]/=len; N[2]/=len; }
    }

    std::sort(result.slices.begin(), result.slices.end(), [&](const SliceMeta &a, const SliceMeta &b){
        if(a.hasPosition && b.hasPosition) {
            double za = a.imagePosition[0]*N[0]+a.imagePosition[1]*N[1]+a.imagePosition[2]*N[2];
            double zb = b.imagePosition[0]*N[0]+b.imagePosition[1]*N[1]+b.imagePosition[2]*N[2];
            return za < zb;
        }
        return a.instanceNumber < b.instanceNumber;
    });

    return result;
}
