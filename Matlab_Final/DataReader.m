function DataReader()

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
    
    % Create time vector (assuming data was collected at regular intervals)
    time = (1:height(data))';
    
    % Plot Accelerometer Data
    figure('Name', 'Accelerometer Data', 'NumberTitle', 'off', 'Position', [50 50 1450 730]);
    
    % Head Accelerometer
    subplot(5,1,1);
    plot(time, Head_Acc);
    ylim([-1.1,1.1]);
    title('Head - Accelerometer');
    xlabel('Sample Number');
    ylabel('Acceleration (g)');
    legend('X', 'Y', 'Z');
    grid on;
    
    % Right Hand Accelerometer
    subplot(5,1,2);
    plot(time, Right_Hand_Acc);
    ylim([-1.1,1.1]);
    title('Right Hand - Accelerometer');
    xlabel('Sample Number');
    ylabel('Acceleration (g)');
    legend('X', 'Y', 'Z');
    grid on;
    
    % Left Hand Accelerometer
    subplot(5,1,3);
    plot(time, Left_Hand_Acc);
    ylim([-1.1,1.1]);
    title('Left Hand - Accelerometer');
    xlabel('Sample Number');
    ylabel('Acceleration (g)');
    legend('X', 'Y', 'Z');
    grid on;

    % Right Foot Accelerometer
    subplot(5,1,4);
    plot(time, Right_Foot_Acc);
    ylim([-1.1,1.1]);
    title('Right Foot - Accelerometer');
    xlabel('Sample Number');
    ylabel('Acceleration (g)');
    legend('X', 'Y', 'Z');
    grid on;
    
    % Left Foot Accelerometer
    subplot(5,1,5);
    plot(time, Left_Foot_Acc);
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
    plot(time, Head_Gyro);
    %ylim([-1.1,1.1]);
    title('Head - Gyroscope');
    xlabel('Sample Number');
    ylabel('Angular Rate (deg/s)');
    legend('X', 'Y', 'Z');
    grid on;
    
    % Right Hand Accelerometer
    subplot(5,1,2);
    plot(time, Right_Hand_Gyro);
    %ylim([-1.1,1.1]);
    title('Right Hand - Gyroscope');
    xlabel('Sample Number');
    ylabel('Angular Rate (deg/s)');
    legend('X', 'Y', 'Z');
    grid on;
    
    % Left Hand Accelerometer
    subplot(5,1,3);
    plot(time, Left_Hand_Gyro);
    %ylim([-1.1,1.1]);
    title('Left Hand - Gyroscope');
    xlabel('Sample Number');
    ylabel('Angular Rate (deg/s)');
    legend('X', 'Y', 'Z');
    grid on;

    % Right Foot Accelerometer
    subplot(5,1,4);
    plot(time, Right_Foot_Gyro);
    %ylim([-1.1,1.1]);
    title('Right Foot - Gyroscope');
    xlabel('Sample Number');
    ylabel('Angular Rate (deg/s)');
    legend('X', 'Y', 'Z');
    grid on;
    
    % Left Foot Accelerometer
    subplot(5,1,5);
    plot(time, Left_Foot_Gyro);
    %ylim([-1.1,1.1]);
    title('Left Foot - Gyroscope');
    xlabel('Sample Number');
    ylabel('Angular Rate (deg/s)');
    legend('X', 'Y', 'Z');
    grid on;
end