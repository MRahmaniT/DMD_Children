clc; clear; close all;

PathName = '/Users/mohammad/University/Bachelor Project/Matlab_Final/FilteredData/';
base = "Filtered.Climb_Box_Right_Foot_2";

for i = 2:26
    FileName = sprintf('%s.%02d.xlsx', base, i);
    DetectAction(PathName, FileName);
end

function DetectAction(pathname, filename)
    % --- Step 1: File selection ---
    fullpath = fullfile(pathname, filename);
    data = readtable(fullpath);

    % --- Step 2: Define parameters ---
    sensors = {'Head', 'RightHand', 'LeftHand', 'RightFoot', 'LeftFoot'};
    fs = 100; % sampling rate in Hz (adjust this to your actual sampling rate)
    baselineDuration = 0.5; % seconds
    baselineSamples = fs * baselineDuration;
    thresholdRatio = 0.03; % 3% change threshold

    % --- Step 3: Prepare storage ---
    startIdxs = zeros(1, numel(sensors));
    endIdxs = zeros(1, numel(sensors));
    acc_mag = zeros(height(data), numel(sensors));

    % --- Step 4: Process each sensor ---
    for i = 1:numel(sensors)
        name = sensors{i};
        ax = data.([name 'Ax']);
        ay = data.([name 'Ay']);
        az = data.([name 'Az']);
        
        % Compute magnitude
        acc_mag(:, i) = sqrt(ax.^2 + ay.^2 + az.^2);

        % Baseline (first 1 second)
        baselineMean = mean(acc_mag(1:baselineSamples, i));

        % Compare deviation to threshold
        diffSignal = abs(acc_mag(:, i) - baselineMean);
        threshold = baselineMean * thresholdRatio;

        % Detect start and end
        startIdx = find(diffSignal > threshold, 1, 'first');
        endIdx = find(diffSignal > threshold, 1, 'last');
        
        if isempty(startIdx)
            startIdx = NaN;
        end
        if isempty(endIdx)
            endIdx = NaN;
        end

        startIdxs(i) = startIdx;
        endIdxs(i) = endIdx;
    end

    % --- Step 5: Combine results ---
    validStartIdxs = startIdxs(~isnan(startIdxs));
    validEndIdxs = endIdxs(~isnan(endIdxs));

    if isempty(validStartIdxs) || isempty(validEndIdxs)
        disp('No motion detected in any sensor.');
        cutData = data;
        path = '/Users/mohammad/University/Bachelor Project/Matlab_Final/DetectedAction/';
        newFile = fullfile(path, ['DetectedAction_' filename]);
        writetable(cutData, newFile);
    end

    globalStart = min(validStartIdxs);
    globalEnd   = max(validEndIdxs);

    % --- Step 6: Cut and save data ---
    cutData = data(globalStart:globalEnd, :);
    path = '/Users/mohammad/University/Bachelor Project/Matlab_Final/DetectedAction/';
    newFile = fullfile(path, ['DetectedAction_' filename]);
    writetable(cutData, newFile);

    disp('✅ Motion detected and segment saved successfully!');
    fprintf('Start Index: %d\nEnd Index: %d\nSaved File: %s\n', ...
        globalStart, globalEnd, newFile);
end
