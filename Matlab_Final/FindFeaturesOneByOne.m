function FindFeaturesOneByOne()
    [filename, pathname] = uigetfile('*.xlsx', 'Select Sensor Data File');
    if isequal(filename, 0)
        disp('User canceled file selection');
        return;
    end
    fullpath = fullfile(pathname, filename);
    
    % Read data from Excel file
    data = readtable(fullpath);

    % Extract data for each sensor
    % Head
    Head_Acc = [data.HeadAx, data.HeadAy, data.HeadAz];
    Head_Gyro = [data.HeadGx, data.HeadGy, data.HeadGz];
    
    % Right Hand
    Right_Hand_Acc = [data.RightHandAx, data.RightHandAy, data.RightHandAz];
    Right_Hand_Gyro = [data.RightHandGx, data.RightHandGy, data.RightHandGz];
    
    % Left Hand
    Left_Hand_Acc = [data.LeftHandAx, data.LeftHandAy, data.LeftHandAz];
    Left_Hand_Gyro = [data.LeftHandGx, data.LeftHandGy, data.LeftHandGz];

    % Right Foot
    Right_Foot_Acc = [data.RightFootAx, data.RightFootAy, data.RightFootAz];
    Right_Foot_Gyro = [data.RightFootGx, data.RightFootGy, data.RightFootGz];
    
    % Left Foot
    Left_Foot_Acc = [data.LeftFootAx, data.LeftFootAy, data.LeftFootAz];
    Left_Foot_Gyro = [data.LeftFootGx, data.LeftFootGy, data.LeftFootGz];
    
    % Find Average
    Head_Acc_Average = mean(Head_Acc);
    Head_Gyro_Average = mean(Head_Gyro);
    
    % Right Hand
    Right_Hand_Acc_Average = mean(Right_Hand_Acc);
    Right_Hand_Gyro_Average = mean(Right_Hand_Gyro);
    
    % Left Hand
    Left_Hand_Acc_Average = mean(Left_Hand_Acc);
    Left_Hand_Gyro_Average = mean(Left_Hand_Gyro);

    % Right Foot
    Right_Foot_Acc_Average = mean(Right_Foot_Acc);
    Right_Foot_Gyro_Average = mean(Right_Foot_Gyro);
    
    % Left Foot
    Left_Foot_Acc_Average = mean(Left_Foot_Acc);
    Left_Foot_Gyro_Average = mean(Left_Foot_Gyro);
    
    fprintf("Head Acceleration          : %d, %d, %d \n" + ...
            "Head Gyroscope             : %d, %d, %d \n" + ...
            "Right Hand Acceleration    : %d, %d, %d \n" + ...
            "Right Hand Gyroscope       : %d, %d, %d \n" + ...
            "Left Hand Acceleration     : %d, %d, %d \n" + ...
            "Left Hand Gyroscope        : %d, %d, %d \n" + ...
            "Right Foot Acceleration    : %d, %d, %d \n" + ...
            "Right Foot Gyroscope       : %d, %d, %d \n" + ...
            "Left Foot Acceleration     : %d, %d, %d \n" + ...
            "Left Foot Gyroscope        : %d, %d, %d \n", ...
            Head_Acc_Average(1), Head_Acc_Average(2), Head_Acc_Average(3), ...
            Head_Gyro_Average(1), Head_Gyro_Average(2), Head_Gyro_Average(3), ...
            Right_Hand_Acc_Average(1), Right_Hand_Acc_Average(2), Right_Hand_Acc_Average(3), ...
            Right_Hand_Gyro_Average(1), Right_Hand_Gyro_Average(2), Right_Hand_Gyro_Average(3), ...
            Left_Hand_Acc_Average(1), Left_Hand_Acc_Average(2), Left_Hand_Acc_Average(3), ...
            Left_Hand_Gyro_Average(1), Left_Hand_Gyro_Average(2), Left_Hand_Gyro_Average(3), ...
            Right_Foot_Acc_Average(1), Right_Foot_Acc_Average(2), Right_Foot_Acc_Average(3), ...
            Right_Foot_Gyro_Average(1), Right_Foot_Gyro_Average(2), Right_Foot_Gyro_Average(3), ...
            Left_Foot_Acc_Average(1), Left_Foot_Acc_Average(2), Left_Foot_Acc_Average(3), ...
            Left_Foot_Gyro_Average(1), Left_Foot_Gyro_Average(2), Left_Foot_Gyro_Average(3));
end