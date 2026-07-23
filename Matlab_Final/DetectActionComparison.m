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

% Two signals to compare
accVariable  = "RightFootAx";
gyroVariable = "RightFootGx";

% Sampling frequency
fs = 100;  % Hz

% Detection parameters
baselineDuration = 0.5;   % Baseline duration in seconds
thresholdRatio   = 0.03;  % 3% difference from baseline

% Plot range in samples
% Use [] to display the complete recording.
displaySampleRange = [0 500];

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
sample = (0:numberOfSamples-1)';

%% ================= CHECK SELECTED SIGNALS =================

variableNames = data.Properties.VariableNames;

if ~ismember(char(accVariable), variableNames)
    error('Accelerometer variable "%s" was not found.', accVariable);
end

if ~ismember(char(gyroVariable), variableNames)
    error('Gyroscope variable "%s" was not found.', gyroVariable);
end

%% ================= EXTRACT TWO SIGNALS =================

accSignal = data.(char(accVariable));
gyroSignal = data.(char(gyroVariable));

accSignal = double(accSignal(:));
gyroSignal = double(gyroSignal(:));

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

    cutData = data(globalStart:globalEnd, :);

    fprintf('\nAction detected successfully.\n');
    fprintf('MATLAB start index: %d\n', globalStart);
    fprintf('MATLAB end index:   %d\n', globalEnd);
    fprintf('Start sample:       %d\n', globalStart - 1);
    fprintf('End sample:         %d\n', globalEnd - 1);
    fprintf('Detected length:    %d samples\n', ...
        globalEnd - globalStart + 1);

else

    warning('No action was detected.');

    globalStart = NaN;
    globalEnd = NaN;

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
    'Position', [100 100 1100 750]);

tiledlayout( ...
    2, 1, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

%% Accelerometer plot

nexttile;

plot( ...
    sample, ...
    accSignal, ...
    'LineWidth', 0.9, ...
    'DisplayName', 'Complete signal');

hold on;

if motionDetected

    detectedSamples = globalStart:globalEnd;

    plot( ...
        sample(detectedSamples), ...
        accSignal(detectedSamples), ...
        'LineWidth', 2, ...
        'DisplayName', 'Detected action');

    xline( ...
        globalStart - 1, ...
        '--', ...
        'Start', ...
        'HandleVisibility', 'off');

    xline( ...
        globalEnd - 1, ...
        '--', ...
        'End', ...
        'HandleVisibility', 'off');
end

grid on;
xlabel('Sample');
ylabel('Acceleration');
title(accVariable + " — Complete and Detected Action");
legend('Location', 'best');

applySampleLimits(displaySampleRange, numberOfSamples);

%% Gyroscope plot

nexttile;

plot( ...
    sample, ...
    gyroSignal, ...
    'LineWidth', 0.9, ...
    'DisplayName', 'Complete signal');

hold on;

if motionDetected

    detectedSamples = globalStart:globalEnd;

    plot( ...
        sample(detectedSamples), ...
        gyroSignal(detectedSamples), ...
        'LineWidth', 2, ...
        'DisplayName', 'Detected action');

    xline( ...
        globalStart - 1, ...
        '--', ...
        'Start', ...
        'HandleVisibility', 'off');

    xline( ...
        globalEnd - 1, ...
        '--', ...
        'End', ...
        'HandleVisibility', 'off');
end

grid on;
xlabel('Sample');
ylabel('Angular velocity');
title(gyroVariable + " — Complete and Detected Action");
legend('Location', 'best');

applySampleLimits(displaySampleRange, numberOfSamples);

if motionDetected
    figureTitle = sprintf( ...
        '%s | Detected samples: %d to %d', ...
        FileName, ...
        globalStart - 1, ...
        globalEnd - 1);
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

function applySampleLimits(displayRange, numberOfSamples)

    if isempty(displayRange)
        xlim([0, max(1, numberOfSamples - 1)]);
        return;
    end

    minimumSample = max(0, displayRange(1));
    maximumSample = min( ...
        displayRange(2), ...
        numberOfSamples - 1);

    if maximumSample <= minimumSample
        maximumSample = max( ...
            minimumSample + 1, ...
            numberOfSamples - 1);
    end

    xlim([minimumSample maximumSample]);
end