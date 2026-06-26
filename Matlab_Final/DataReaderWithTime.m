function DataReaderWithTime()

    % Prompt user to select the Excel file
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
    Head_Time = data.HeadTime;
    Head_Acc = [data.HeadAx, data.HeadAy, data.HeadAz];
    Head_Gyro = [data.HeadGx, data.HeadGy, data.HeadGz];
    
    % Right Hand
    Right_Hand_Time = data.RightHandTime;
    Right_Hand_Acc = [data.RightHandAx, data.RightHandAy, data.RightHandAz];
    Right_Hand_Gyro = [data.RightHandGx, data.RightHandGy, data.RightHandGz];
    
    % Left Hand
    Left_Hand_Time = data.LeftHandTime;
    Left_Hand_Acc = [data.LeftHandAx, data.LeftHandAy, data.LeftHandAz];
    Left_Hand_Gyro = [data.LeftHandGx, data.LeftHandGy, data.LeftHandGz];

    % Right Foot
    Right_Foot_Time = data.RightFootTime;
    Right_Foot_Acc = [data.RightFootAx, data.RightFootAy, data.RightFootAz];
    Right_Foot_Gyro = [data.RightFootGx, data.RightFootGy, data.RightFootGz];
    
    % Left Foot
    Left_Foot_Time = data.LeftFootTime;
    Left_Foot_Acc = [data.LeftFootAx, data.LeftFootAy, data.LeftFootAz];
    Left_Foot_Gyro = [data.LeftFootGx, data.LeftFootGy, data.LeftFootGz];
    
    % Plot Accelerometer Data
    figure('Name', 'Accelerometer Data', 'NumberTitle', 'off', 'Position', [50 50 1450 730]);
    
    % Head Accelerometer
    subplot(5,1,1);
    plot(Head_Time, Head_Acc);
    ylim([-1.1,1.1]);
    title('Head - Accelerometer');
    xlabel('Sample Number');
    ylabel('Acceleration (g)');
    legend('X', 'Y', 'Z');
    grid on;
    
    % Right Hand Accelerometer
    subplot(5,1,2);
    plot(Right_Hand_Time, Right_Hand_Acc);
    ylim([-1.1,1.1]);
    title('Right Hand - Accelerometer');
    xlabel('Sample Number');
    ylabel('Acceleration (g)');
    legend('X', 'Y', 'Z');
    grid on;
    
    % Left Hand Accelerometer
    subplot(5,1,3);
    plot(Left_Hand_Time, Left_Hand_Acc);
    ylim([-1.1,1.1]);
    title('Left Hand - Accelerometer');
    xlabel('Sample Number');
    ylabel('Acceleration (g)');
    legend('X', 'Y', 'Z');
    grid on;

    % Right Foot Accelerometer
    subplot(5,1,4);
    plot(Right_Foot_Time, Right_Foot_Acc);
    ylim([-1.1,1.1]);
    title('Right Foot - Accelerometer');
    xlabel('Sample Number');
    ylabel('Acceleration (g)');
    legend('X', 'Y', 'Z');
    grid on;
    
    % Left Foot Accelerometer
    subplot(5,1,5);
    plot(Left_Foot_Time, Left_Foot_Acc);
    ylim([-1.1,1.1]);
    title('Left Foot - Accelerometer');
    xlabel('Sample Number');
    ylabel('Acceleration (g)');
    legend('X', 'Y', 'Z');
    grid on;
    
    % Plot Gyroscope Data
    figure('Name', 'Gyroscope Data', 'NumberTitle', 'off', 'Position', [50 50 1450 730]);
    
    % Head Gyroscope
    subplot(5,1,1);
    plot(Head_Time, Head_Gyro);
    %ylim([-1.1,1.1]);
    title('Head - Gyroscope');
    xlabel('Sample Number');
    ylabel('Angular Rate (deg/s)');
    legend('X', 'Y', 'Z');
    grid on;
    
    % Right Hand Accelerometer
    subplot(5,1,2);
    plot(Right_Hand_Time, Right_Hand_Gyro);
    %ylim([-1.1,1.1]);
    title('Right Hand - Gyroscope');
    xlabel('Sample Number');
    ylabel('Angular Rate (deg/s)');
    legend('X', 'Y', 'Z');
    grid on;
    
    % Left Hand Accelerometer
    subplot(5,1,3);
    plot(Left_Hand_Time, Left_Hand_Gyro);
    %ylim([-1.1,1.1]);
    title('Left Hand - Gyroscope');
    xlabel('Sample Number');
    ylabel('Angular Rate (deg/s)');
    legend('X', 'Y', 'Z');
    grid on;

    % Right Foot Accelerometer
    subplot(5,1,4);
    plot(Right_Foot_Time, Right_Foot_Gyro);
    %ylim([-1.1,1.1]);
    title('Right Foot - Gyroscope');
    xlabel('Sample Number');
    ylabel('Angular Rate (deg/s)');
    legend('X', 'Y', 'Z');
    grid on;
    
    % Left Foot Accelerometer
    subplot(5,1,5);
    plot(Left_Foot_Time, Left_Foot_Gyro);
    %ylim([-1.1,1.1]);
    title('Left Foot - Gyroscope');
    xlabel('Sample Number');
    ylabel('Angular Rate (deg/s)');
    legend('X', 'Y', 'Z');
    grid on;
end