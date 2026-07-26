clc;
clear;
close all;

%% ================= CHANGEABLE PARAMETERS =================

% Input filtered-data folder
PathName = ...
    '/Users/mohammad/University/Bachelor Project/Matlab_Final/FilteredData/';

% Output folders
detectedDataFolder = ...
    '/Users/mohammad/University/Bachelor Project/Matlab_Final/DetectedAction/';

plotFolder = ...
    '/Users/mohammad/University/Bachelor Project/Matlab_Final/Plots/';

% Select one action and trial
base = "Filtered.Sit_To_Stand_2";
fileNumber = 2;

% Signals to plot
timeVariable = "RightFootTime";
accVariable  = "RightFootAx";
gyroVariable = "RightFootGx";

% Sampling frequency
fs = 100;  % Hz

% Detection parameters
baselineDuration = 0.5;   % Baseline duration in seconds
thresholdRatio   = 0.015;  % 3% difference from baseline

% Plot range in seconds
% Use [] to display the complete recording.
displayTimeRange = [0 5];

% Save settings
saveDetectedData = true;
savePNG = true;
savePDF = true;
saveSVG = true;

%% ================= READ ONE ACTION FILE =================

FileName = sprintf('%s.%02d.xlsx', base, fileNumber);
fullPath = fullfile(PathName, FileName);

if ~isfile(fullPath)
    error('File not found:\n%s', fullPath);
end

data = readtable(fullPath);

numberOfSamples = height(data);

if numberOfSamples == 0
    error('The selected file contains no data.');
end

% Sample numbers starting from zero
sample = (0:numberOfSamples - 1)';

%% ================= CHECK SELECTED SIGNALS =================

variableNames = data.Properties.VariableNames;

if ~ismember(char(timeVariable), variableNames)
    error('Time variable "%s" was not found.', timeVariable);
end

if ~ismember(char(accVariable), variableNames)
    error('Accelerometer variable "%s" was not found.', accVariable);
end

if ~ismember(char(gyroVariable), variableNames)
    error('Gyroscope variable "%s" was not found.', gyroVariable);
end

%% ================= EXTRACT TIME AND SIGNALS =================

% Convert time from milliseconds to seconds and make it start from zero
time = double(data.(char(timeVariable)));
time = time(:);

time = time / 1000;
time = time - time(1);

accSignal = double(data.(char(accVariable)));
gyroSignal = double(data.(char(gyroVariable)));

accSignal = accSignal(:);
gyroSignal = gyroSignal(:);

% The input file is already filtered and its gyroscope bias was already
% removed in the filtering code. Therefore, do not subtract the bias again.

%% ================= ACTION DETECTION =================

sensors = { ...
    'Head', ...
    'RightHand', ...
    'LeftHand', ...
    'RightFoot', ...
    'LeftFoot'};

numberOfSensors = numel(sensors);

baselineSamples = round(fs * baselineDuration);
baselineSamples = max(1, baselineSamples);
baselineSamples = min(baselineSamples, numberOfSamples);

startIdxs = nan(1, numberOfSensors);
endIdxs = nan(1, numberOfSensors);

accMagnitude = nan(numberOfSamples, numberOfSensors);

for sensorIndex = 1:numberOfSensors

    sensorName = sensors{sensorIndex};

    axName = [sensorName 'Ax'];
    ayName = [sensorName 'Ay'];
    azName = [sensorName 'Az'];

    % Check sensor columns
    requiredVariables = {axName, ayName, azName};

    if ~all(ismember(requiredVariables, variableNames))
        warning('Acceleration columns for %s were not found.', sensorName);
        continue;
    end

    ax = double(data.(axName));
    ay = double(data.(ayName));
    az = double(data.(azName));

    ax = ax(:);
    ay = ay(:);
    az = az(:);

    % Acceleration magnitude
    accMagnitude(:, sensorIndex) = sqrt( ...
        ax.^2 + ...
        ay.^2 + ...
        az.^2);

    % Baseline from the beginning of the signal
    baselineMean = mean( ...
        accMagnitude(1:baselineSamples, sensorIndex), ...
        'omitnan');

    % Difference from the baseline
    differenceSignal = abs( ...
        accMagnitude(:, sensorIndex) - baselineMean);

    % Detection threshold
    threshold = abs(baselineMean) * thresholdRatio;

    % Samples above the threshold
    activeSamples = differenceSignal > threshold;

    % Baseline samples are used for calibration, not detection
    activeSamples(1:baselineSamples) = false;

    % Find first and last active sample
    startIndex = find(activeSamples, 1, 'first');
    endIndex = find(activeSamples, 1, 'last');

    if ~isempty(startIndex)
        startIdxs(sensorIndex) = startIndex;
    end

    if ~isempty(endIndex)
        endIdxs(sensorIndex) = endIndex;
    end
end

%% ================= COMBINE SENSOR RESULTS =================

validStartIdxs = startIdxs(~isnan(startIdxs));
validEndIdxs = endIdxs(~isnan(endIdxs));

motionDetected = ...
    ~isempty(validStartIdxs) && ...
    ~isempty(validEndIdxs);

if motionDetected

    globalStart = min(validStartIdxs);
    globalEnd = max(validEndIdxs);

    % Corresponding detection times
    startTime = time(globalStart);
    endTime = time(globalEnd);

    cutData = data(globalStart:globalEnd, :);

    fprintf('\nAction detected successfully.\n');
    fprintf('MATLAB start index: %d\n', globalStart);
    fprintf('MATLAB end index:   %d\n', globalEnd);
    fprintf('Start sample:       %d\n', globalStart - 1);
    fprintf('End sample:         %d\n', globalEnd - 1);
    fprintf('Start time:         %.3f seconds\n', startTime);
    fprintf('End time:           %.3f seconds\n', endTime);
    fprintf('Action duration:    %.3f seconds\n', endTime - startTime);
    fprintf('Detected length:    %d samples\n', ...
        globalEnd - globalStart + 1);

else

    warning('No action was detected.');

    globalStart = NaN;
    globalEnd = NaN;

    startTime = NaN;
    endTime = NaN;

    % Preserve complete data if no action is detected
    cutData = data;
end

%% ================= SAVE DETECTED DATA =================

if saveDetectedData

    if ~exist(detectedDataFolder, 'dir')
        mkdir(detectedDataFolder);
    end

    detectedFileName = ['DetectedAction_' FileName];

    detectedFilePath = fullfile( ...
        detectedDataFolder, ...
        detectedFileName);

    writetable(cutData, detectedFilePath);

    fprintf('Detected data saved:\n%s\n', detectedFilePath);
end

%% ================= PLOT TWO SIGNALS =================

fig = figure( ...
    'Color', 'white', ...
    'Name', 'Action Detection');

tiledlayout( ...
    2, 1, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

%% ================= ACCELEROMETER PLOT =================

nexttile;

plot( ...
    time, ...
    accSignal, ...
    'LineWidth', 0.9, ...
    'DisplayName', 'Complete signal');

hold on;

if motionDetected

    detectedIndices = globalStart:globalEnd;

    plot( ...
        time(detectedIndices), ...
        accSignal(detectedIndices), ...
        'LineWidth', 2, ...
        'DisplayName', 'Detected action');

    % Start line based on time, not sample index
    xline( ...
        startTime, ...
        '--', ...
        sprintf('Start: %.2f s', startTime), ...
        'LineWidth', 1.2, ...
        'LabelVerticalAlignment', 'bottom', ...
        'HandleVisibility', 'off');

    % End line based on time, not sample index
    xline( ...
        endTime, ...
        '--', ...
        sprintf('End: %.2f s', endTime), ...
        'LineWidth', 1.2, ...
        'LabelVerticalAlignment', 'bottom', ...
        'HandleVisibility', 'off');
end

grid on;

xlabel('Time (s)');
ylabel('Acceleration (g)');

title( ...
    accVariable + " — Complete and Detected Action", ...
    'Interpreter', 'none');

legend('Location', 'northeast');

applyTimeLimits(displayTimeRange, time);

%% ================= GYROSCOPE PLOT =================

nexttile;

plot( ...
    time, ...
    gyroSignal, ...
    'LineWidth', 0.9, ...
    'DisplayName', 'Complete signal');

hold on;

if motionDetected

    detectedIndices = globalStart:globalEnd;

    plot( ...
        time(detectedIndices), ...
        gyroSignal(detectedIndices), ...
        'LineWidth', 2, ...
        'DisplayName', 'Detected action');

    % Start line based on time, not sample index
    xline( ...
        startTime, ...
        '--', ...
        sprintf('Start: %.2f s', startTime), ...
        'LineWidth', 1.2, ...
        'LabelVerticalAlignment', 'bottom', ...
        'HandleVisibility', 'off');

    % End line based on time, not sample index
    xline( ...
        endTime, ...
        '--', ...
        sprintf('End: %.2f s', endTime), ...
        'LineWidth', 1.2, ...
        'LabelVerticalAlignment', 'bottom', ...
        'HandleVisibility', 'off');
end

grid on;

xlabel('Time (s)');
ylabel('Angular velocity (deg/s)');

title( ...
    gyroVariable + " — Complete and Detected Action", ...
    'Interpreter', 'none');

legend('Location', 'northeast');

applyTimeLimits(displayTimeRange, time);

%% ================= FIGURE TITLE =================

if motionDetected

    figureTitle = sprintf( ...
        '%s | Detected time: %.2f to %.2f s | Duration: %.2f s', ...
        FileName, ...
        startTime, ...
        endTime, ...
        endTime - startTime);

else

    figureTitle = sprintf( ...
        '%s | No action detected', ...
        FileName);
end

sgtitle(figureTitle, 'Interpreter', 'none');

%% ================= SAVE PLOT =================

if ~exist(plotFolder, 'dir')
    mkdir(plotFolder);
end

plotName = sprintf( ...
    '%s.%02d_%s_%s_ActionDetection', ...
    base, ...
    fileNumber, ...
    accVariable, ...
    gyroVariable);

if savePNG

    pngPath = fullfile( ...
        plotFolder, ...
        plotName + ".png");

    exportgraphics( ...
        fig, ...
        pngPath, ...
        'Resolution', 300);

    fprintf('PNG saved:\n%s\n', pngPath);
end

if savePDF

    pdfPath = fullfile( ...
        plotFolder, ...
        plotName + ".pdf");

    exportgraphics( ...
        fig, ...
        pdfPath, ...
        'ContentType', 'vector');

    fprintf('PDF saved:\n%s\n', pdfPath);
end

if saveSVG

    svgPath = fullfile( ...
        plotFolder, ...
        plotName + ".svg");

    % Compatible with older MATLAB versions
    print( ...
        fig, ...
        svgPath, ...
        '-dsvg', ...
        '-painters');

    fprintf('SVG saved:\n%s\n', svgPath);
end

%% ================= LOCAL FUNCTION =================

function applyTimeLimits(displayRange, time)

    % Remove invalid time values when determining the valid range
    validTime = time(isfinite(time));

    if isempty(validTime)
        warning('No valid time values were found.');
        return;
    end

    recordingStartTime = validTime(1);
    recordingEndTime = validTime(end);

    % Display the complete recording
    if isempty(displayRange)

        if recordingEndTime > recordingStartTime
            xlim([recordingStartTime, recordingEndTime]);
        end

        return;
    end

    % Make sure the display range contains two values
    if numel(displayRange) ~= 2
        warning( ...
            ['displayTimeRange must contain two values. ', ...
             'The complete recording will be displayed.']);

        if recordingEndTime > recordingStartTime
            xlim([recordingStartTime, recordingEndTime]);
        end

        return;
    end

    minimumTime = max(recordingStartTime, displayRange(1));
    maximumTime = min(recordingEndTime, displayRange(2));

    if maximumTime <= minimumTime

        warning( ...
            ['The selected display time range is invalid. ', ...
             'The complete recording will be displayed.']);

        if recordingEndTime > recordingStartTime
            xlim([recordingStartTime, recordingEndTime]);
        end

        return;
    end

    xlim([minimumTime, maximumTime]);
end