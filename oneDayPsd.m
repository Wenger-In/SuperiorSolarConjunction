function [NOFILE,DATALACK,freq,pxx,Freqs,papoulis_psdx] = oneDayPsd(year_str,month_str,day_str,dataDir_str,winFactor)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明


NOFILE = 0;
DATALACK = 0;  %0表示正常，数据缺失小于1%
%% load data for 1 day
fileName = [dataDir_str year_str '\psp_fld_l2_mag_rtn_' year_str month_str day_str '*_*.cdf'];
files = dir(fileName);
if isempty(files)
    NOFILE = 1
    freq = [];
    pxx = [];
    Freqs = [];
    papoulis_psdx = [];
    return
end
cdfinfo = 0;
cdfVariables = 0;
epoch = [];
mag_rtn = [];
files_founded = length(files)
for file_i = 1:files_founded
    thefileName = [dataDir_str year_str '\' files(file_i).name];

    if file_i==1
        cdfinfo = spdfcdfinfo(thefileName);
        cdfVariables = cdfinfo.Variables(:,1);
    end
    cdfdata = spdfcdfread(thefileName);
    epoch = [epoch; cdfdata{strcmp(cdfVariables,'epoch_mag_RTN')}];
    mag_rtn = [mag_rtn; cdfdata{strcmp(cdfVariables,'psp_fld_l2_mag_RTN')}];
end

%% 检查NAN数据量，如果缺失较少，则线性插值
nanpoints = sum(isnan(mag_rtn),2)>=1;
nan_count = sum(nanpoints);
if nan_count>(length(mag_rtn)/100)
    DATALACK = 1
    freq = [];
    pxx = [];
    Freqs = [];
    papoulis_psdx = [];
    return
else
    mag_rtn(nanpoints,:) = interp1(epoch(~nanpoints),mag_rtn(~nanpoints,:),epoch(nanpoints),'linear');
end
% 边缘的NAN就直接忽略了
nanpoints = sum(isnan(mag_rtn),2)>=1;
mag_rtn(nanpoints,:) = [];
epoch(nanpoints) = [];

%% 检查这天中数据是否有较大的缺失
MinPeriod = min(diff(epoch))*24*3600;                   % Sampling Period (s)
MaxSampleGap = max(diff(epoch))*24*3600;   
if (MaxSampleGap/MinPeriod) > (length(mag_rtn)/100)
    DATALACK = 1
    freq = [];
    pxx = [];
    Freqs = [];
    papoulis_psdx = [];
    return
end

%% 计算这一天的PSD
SamplePeriod = mean(diff(epoch))*24*3600;                   % Sampling Period (s)
Fs = 1/SamplePeriod;                                        % Sampling Frequency (Hz)
dmag_rtn = mag_rtn - mean(mag_rtn,1,"omitnan");             % Remove d-c Offset
dfreq = Fs/length(dmag_rtn)/2;
freq = 0:dfreq:Fs/2;
pxx = periodogram(dmag_rtn,[],2*length(dmag_rtn));
pxx = sum(pxx,2);

pointsFactor = 1.0005;
freqnum = floor(log(length(freq))/log(pointsFactor));
Freq_is = ceil(pointsFactor.^(0:freqnum));
Freq_is = unique(Freq_is);
length(Freq_is)
Freqs = freq(Freq_is);
papoulis_psdx = zeros(size(Freq_is));
i=1;
for thefi = Freq_is
    theNw = freq(thefi)/winFactor/dfreq;
    theNw = floor(theNw);
    if mod(theNw,2)==0
        theNw = theNw + 1;
    end
    thewin = papouliswin(theNw)/theNw;
    papoulis_psdx(i) = logarithmicSmoothing(pxx,thewin,thefi);
    i = i + 1;
end

end

