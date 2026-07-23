clc; clear; close all;

PathName = '/Users/mohammad/University/Bachelor Project/Matlab_Final/Data/';
base = "Sit_To_Stand_2";

for i = 1:26
    FileName = sprintf('%s.%02d.xlsx', base, i);
    FilterData(PathName, FileName);
end

function FilterData(pathname, filename)
    %filtering defaults (can be overridden by UI at save time)
    fs_nominal   = 100;     % for filter design
    filterOrder  = 4;
    fc_lp        = 4;       % Hz low-pass (can change in UI)

    fullpath = fullfile(pathname, filename);
    % Read data from Excel file
    data = readtable(fullpath);

    % Extract data for each sensor
    % Head
    Head_Time = data.HeadTime;
    Head_Time = Head_Time(:) - Head_Time(1);
    Head_Acc = [data.HeadAx, data.HeadAy, data.HeadAz];
    Head_Gyro = [data.HeadGx, data.HeadGy, data.HeadGz];
    
    % Right Hand
    Right_Hand_Time = data.RightHandTime;
    Right_Hand_Time = Right_Hand_Time(:) - Right_Hand_Time(1);
    Right_Hand_Acc = [data.RightHandAx, data.RightHandAy, data.RightHandAz];
    Right_Hand_Gyro = [data.RightHandGx, data.RightHandGy, data.RightHandGz];
    
    % Left Hand
    Left_Hand_Time = data.LeftHandTime;
    Left_Hand_Time = Left_Hand_Time(:) - Left_Hand_Time(1);
    Left_Hand_Acc = [data.LeftHandAx, data.LeftHandAy, data.LeftHandAz];
    Left_Hand_Gyro = [data.LeftHandGx, data.LeftHandGy, data.LeftHandGz];

    % Right Foot
    Right_Foot_Time = data.RightFootTime;
    Right_Foot_Time = Right_Foot_Time(:) - Right_Foot_Time(1);
    Right_Foot_Acc = [data.RightFootAx, data.RightFootAy, data.RightFootAz];
    Right_Foot_Gyro = [data.RightFootGx, data.RightFootGy, data.RightFootGz];
    
    % Left Foot
    Left_Foot_Time = data.LeftFootTime;
    Left_Foot_Time = Left_Foot_Time(:) - Left_Foot_Time(1);
    Left_Foot_Acc = [data.LeftFootAx, data.LeftFootAy, data.LeftFootAz];
    Left_Foot_Gyro = [data.LeftFootGx, data.LeftFootGy, data.LeftFootGz];
        
    % Bias
    Head_Gyroscope_Bias = [1.561420e-01, -1.808080e-01, 1.249570e-01];
    Right_Hand_Gyroscope_Bias = [5.235010e-01, -6.260000e-03, 2.401080e-01];
    Left_Hand_Gyroscope_Bias = [-1.943000e-03, 1.797440e-01, 1.992370e-01];
    Right_Foot_Gyroscope_Bias = [-6.606800e-02, -1.118280e-01, -3.780940e-01];
    Left_Foot_Gyroscope_Bias = [-8.873500e-02, -7.624300e-02, -2.763200e-01];

    % Filtered Data
    Head_Gyro(:,1) = Head_Gyro(:,1) - Head_Gyroscope_Bias(1);
    Head_Gyro(:,2) = Head_Gyro(:,2) - Head_Gyroscope_Bias(2);
    Head_Gyro(:,3) = Head_Gyro(:,3) - Head_Gyroscope_Bias(3); 
    Right_Hand_Gyro(:,1) = Right_Hand_Gyro(:,1) - Right_Hand_Gyroscope_Bias(1);
    Right_Hand_Gyro(:,2) = Right_Hand_Gyro(:,2) - Right_Hand_Gyroscope_Bias(2);
    Right_Hand_Gyro(:,3) = Right_Hand_Gyro(:,3) - Right_Hand_Gyroscope_Bias(3); 
    Left_Hand_Gyro(:,1)  = Left_Hand_Gyro(:,1) - Left_Hand_Gyroscope_Bias(1);
    Left_Hand_Gyro(:,2) = Left_Hand_Gyro(:,2) - Left_Hand_Gyroscope_Bias(2);
    Left_Hand_Gyro(:,3) = Left_Hand_Gyro(:,3) - Left_Hand_Gyroscope_Bias(3); 
    Right_Foot_Gyro(:,1) = Right_Foot_Gyro(:,1) - Right_Foot_Gyroscope_Bias(1);
    Right_Foot_Gyro(:,2) = Right_Foot_Gyro(:,2) - Right_Foot_Gyroscope_Bias(2);
    Right_Foot_Gyro(:,3) = Right_Foot_Gyro(:,3) - Right_Foot_Gyroscope_Bias(3); 
    Left_Foot_Gyro(:,1)  = Left_Foot_Gyro(:,1) - Left_Foot_Gyroscope_Bias(1);
    Left_Foot_Gyro(:,2) = Left_Foot_Gyro(:,2) - Left_Foot_Gyroscope_Bias(2);
    Left_Foot_Gyro(:,3) = Left_Foot_Gyro(:,3) - Left_Foot_Gyroscope_Bias(3); 

    % Ensure valid numeric filter params
    fc = fc_lp;
    ord = filterOrder;
    fs = fs_nominal;
    if ~isfinite(fc) || isnan(fc) || fc<=0, fc = 5; end
    nyq = fs/2;
    fc = min(max(fc, 0.01), nyq*0.95);  % clamp between 0.01 Hz and 95% Nyquist
    if ~isfinite(ord) || isnan(ord) || ord<1, ord = 4; end
    ord = max(1, round(ord));
    fcSafe = fc; ordSafe = ord;

    [b,a] = butter(ordSafe, fcSafe/(fs_nominal/2), 'low');

    Head_Acc = filtfilt(b,a,Head_Acc);
    Head_Gyro = filtfilt(b,a,Head_Gyro);
    Right_Hand_Acc   = filtfilt(b,a,Right_Hand_Acc);  
    Right_Hand_Gyro   = filtfilt(b,a,Right_Hand_Gyro);  
    Left_Hand_Acc   = filtfilt(b,a,Left_Hand_Acc);  
    Left_Hand_Gyro   = filtfilt(b,a,Left_Hand_Gyro);  
    Right_Foot_Acc   = filtfilt(b,a,Right_Foot_Acc);  
    Right_Foot_Gyro   = filtfilt(b,a,Right_Foot_Gyro);  
    Left_Foot_Acc   = filtfilt(b,a,Left_Foot_Acc);  
    Left_Foot_Gyro   = filtfilt(b,a,Left_Foot_Gyro); 

    M = [ Head_Time, Head_Acc, Head_Gyro, ...
      Right_Hand_Time, Right_Hand_Acc,   Right_Hand_Gyro,   ...
      Left_Hand_Time, Left_Hand_Acc,   Left_Hand_Gyro,   ...
      Right_Foot_Time, Right_Foot_Acc,   Right_Foot_Gyro,   ...
      Left_Foot_Time, Left_Foot_Acc,   Left_Foot_Gyro ];
    
    varNames = { ...
    'Head Time', 'Head Ax','Head Ay','Head Az','Head Gx','Head Gy','Head Gz', ...
    'Right Hand Time', 'Right Hand Ax','Right Hand Ay','Right Hand Az','Right Hand Gx','Right Hand Gy','Right Hand Gz', ...
    'Left Hand Time', 'Left Hand Ax','Left Hand Ay','Left Hand Az','Left Hand Gx','Left Hand Gy','Left Hand Gz', ...
    'Right Foot Time', 'Right Foot Ax','Right Foot Ay','Right Foot Az','Right Foot Gx','Right Foot Gy','Right Foot Gz', ...
    'Left Foot Time', 'Left Foot Ax','Left Foot Ay','Left Foot Az','Left Foot Gx','Left Foot Gy','Left Foot Gz' };


    T = array2table(M, 'VariableNames', varNames);

    filePath = fullfile("FilteredData", "Filtered." + filename);
    writetable(T, filePath);
end
