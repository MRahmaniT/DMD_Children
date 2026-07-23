clc;
clear;
close all;

%% ================= CHANGEABLE PARAMETERS =================

PathName = '/Users/mohammad/University/Bachelor Project/Matlab_Final/Data/';

base       = "Sit_To_Stand_2";
% base       = "Climb_Box_Right_Foot_2";

Title = "Action: Sit To Stand (Score 2)";

fileNumber = 2;     % Action/trial number

% Select sensor signals
timeVariable = "RightFootTime";

accVariable  = "RightFootAx";   % One accelerometer signal
gyroVariable = "RightFootGx";   % One gyroscope signal

% Gyroscope bias
applyGyroBias = true;
gyroBias      = -6.606800e-02;   % RightFootGx bias

% Filter parameters
fs          = 100;   % Sampling frequency, Hz
filterOrder = 4;
cutoffFreq  = 4;     % Cutoff frequency, Hz

% Save plot settings
savePlot   = true;
savePNG    = true;
savePDF    = true;
saveSVG = true;

outputFolder = fullfile('/Users/mohammad/University/Bachelor Project/Matlab_Final/Plots');

%% ================= READ ONE ACTION FILE =================

FileName = sprintf('%s.%02d.xlsx', base, fileNumber);
fullPath = fullfile(PathName, FileName);

if ~isfile(fullPath)
    error('File not found: %s', fullPath);
end

data = readtable(fullPath);

%% ================= EXTRACT TWO SIGNALS =================

time = data.(timeVariable);
time = time(:)/1000 - time(1)/1000;

accRaw  = data.(accVariable);
gyroRaw = data.(gyroVariable);

accRaw  = accRaw(:);
gyroRaw = gyroRaw(:);

% Remove gyroscope bias
if applyGyroBias
    gyroRaw = gyroRaw - gyroBias;
end

%% ================= DESIGN LOW-PASS FILTER =================

nyquistFreq = fs / 2;

if cutoffFreq <= 0 || cutoffFreq >= nyquistFreq
    error('Cutoff frequency must be between 0 and %.2f Hz.', nyquistFreq);
end

[b, a] = butter( ...
    filterOrder, ...
    cutoffFreq / nyquistFreq, ...
    'low');

%% ================= FILTER TWO SIGNALS =================

accFiltered  = filtfilt(b, a, accRaw);
gyroFiltered = filtfilt(b, a, gyroRaw);

%% ================= PLOT COMPARISON =================

fig = figure('Color', 'white');

tiledlayout(2, 1, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

% Accelerometer
nexttile;

plot(time, accRaw, ...
    'DisplayName', 'Before filter');

hold on;

plot(time, accFiltered, ...
    'LineWidth', 1.5, ...
    'DisplayName', 'After filter');

grid on;
xlabel('Time (s)');
ylabel('Acceleration (g)');
title(accVariable + ": Before and After Filtering");
legend('Location', 'best');

% Gyroscope
nexttile;

plot(time, gyroRaw, ...
    'DisplayName', 'Before filter');

hold on;

plot(time, gyroFiltered, ...
    'LineWidth', 1.5, ...
    'DisplayName', 'After filter');

grid on;
xlabel('Time (s)');
ylabel('Angular velocity (deg/s)');
title(gyroVariable + ": Before and After Filtering");
legend('Location', 'best');

sgtitle(Title);

%% ================= SAVE PLOT =================

if savePlot

    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end

    plotName = sprintf('%s.%02d_%s_%s_FilterComparison', ...
        base, ...
        fileNumber, ...
        accVariable, ...
        gyroVariable);

    if savePNG
        pngPath = fullfile(outputFolder, plotName + ".png");

        exportgraphics(fig, pngPath, ...
            'Resolution', 300);

        fprintf('PNG saved:\n%s\n', pngPath);
    end

    if savePDF
        pdfPath = fullfile(outputFolder, plotName + ".pdf");

        exportgraphics(fig, pdfPath, ...
            'ContentType', 'vector');

        fprintf('PDF saved:\n%s\n', pdfPath);
    end

    if saveSVG
        svgPath = fullfile(outputFolder, plotName + ".svg");

        print( ...
        fig, ...
        svgPath, ...
        '-dsvg', ...
        '-painters');

        fprintf('SVG saved:\n%s\n', svgPath);
    end
end