function DetectActionOne()
    % --- Step 1: File selection ---
    [filename, pathname] = uigetfile('*.xlsx', 'Select Sensor Data File');
    if isequal(filename, 0)
        disp('User canceled file selection');
        return;
    end
    fullpath = fullfile(pathname, filename);
    data = readtable(fullpath);

    % --- Step 2: Define parameters ---
    sensors = {'Head', 'RightHand', 'LeftHand', 'RightFoot', 'LeftFoot'};
    fs = 100; % sampling rate in Hz (adjust this to your actual sampling rate)
    baselineDuration = 1; % seconds
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

        % Optional visualization
        figure('Name', [name ' Detection'], 'NumberTitle', 'off');
        plot(acc_mag(:, i)); hold on;
        yline(baselineMean, '--r', 'Baseline');
        yline(baselineMean + threshold, ':k', '+10%');
        yline(baselineMean - threshold, ':k', '-10%');
        if ~isnan(startIdx), xline(startIdx, '--g', 'Start'); end
        if ~isnan(endIdx), xline(endIdx, '--m', 'End'); end
        xlabel('Sample'); ylabel('Acceleration Magnitude');
        title([name ' - Motion Detection']);
        grid on;
    end

    % --- Step 5: Combine results ---
    validStartIdxs = startIdxs(~isnan(startIdxs));
    validEndIdxs = endIdxs(~isnan(endIdxs));

    if isempty(validStartIdxs) || isempty(validEndIdxs)
        disp('No motion detected in any sensor.');
        return;
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

    % --- Step 7: Optional - Overall visualization ---
    figure('Name', 'Combined Action Detection', 'NumberTitle', 'off');
    plot(mean(acc_mag, 2), 'b'); hold on;
    xline(globalStart, '--g', 'Global Start');
    xline(globalEnd, '--m', 'Global End');
    title('Combined Sensor Activity');
    xlabel('Sample Number'); ylabel('Mean Acceleration Magnitude');
    grid on;
end
