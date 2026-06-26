function FindFeaturesAllTogether()
    clc;
    clear;
    close all;
    data1 = readtable("Bias/Bias.01.xlsx");
    data2 = readtable("Bias/Bias.02.xlsx");
    data3 = readtable("Bias/Bias.03.xlsx");
    data4 = readtable("Bias/Bias.04.xlsx");
    data5 = readtable("Bias/Bias.05.xlsx");
    data6 = readtable("Bias/Bias2.01.xlsx");
    data7 = readtable("Bias/Bias2.02.xlsx");
    data8 = readtable("Bias/Bias2.03.xlsx");
    data9 = readtable("Bias/Bias2.04.xlsx");
    data10 = readtable("Bias/Bias2.05.xlsx");

    data = [data1; data2; data3; data4; data5; data6; data7; data8; data9; data10];


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