%% Five-sensor binary reader (MPU packets) + realtime plot + CSV save
clear; clc;

% --------- Sensors ----------
S(1).name = "TTGO-06"; S(1).tag = "Head";
S(2).name = "TTGO-11"; S(2).tag = "Right_Hand";
S(3).name = "TTGO-12"; S(3).tag = "Left_Hand";
S(4).name = "TTGO-18"; S(4).tag = "Right_Foot";
S(5).name = "TTGO-19"; S(5).tag = "Left_Foot";

% Bluetooth Chanel
CH      = 1;

% Data Condition
PKT_LEN = 20;
SYNC1   = uint8(hex2dec('55'));
SYNC2   = uint8(hex2dec('AA'));

% Sensor Scale
ACCEL_SCALE = 8192.0; % LSB/g (example: +/-4g)
GYRO_SCALE  = 16.4;   % LSB/(deg/s) (example: +/-2000 dps)

% --------- Storage ----------
for i = 1:numel(S)
    S(i).bt = [];
    S(i).buf = uint8([]);
    S(i).lastSeq = [];
    S(i).totalBytes = 0;
    S(i).pktCount = 0;
    % store: [seq time_ms ax_g ay_g az_g gx_dps gy_dps gz_dps]
    S(i).dataMat = zeros(0,8);
end

% --------- Connect ----------
fprintf("Connecting to 5 sensors on channel %d...\n", CH);
for i = 1:numel(S)
    fprintf("  [%s] Connecting to %s ... ", S(i).tag, S(i).name);
    try
        S(i).bt = bluetooth(S(i).name, CH);
        configureTerminator(S(i).bt, "LF"); % keep (your MATLAB requirement)
        flush(S(i).bt);
        fprintf("OK\n");
    catch ME
        fprintf("FAILED: %s\n", ME.message);
        S(i).bt = [];
    end
end

% If any failed, still run with the ones that connected
isOn = arrayfun(@(x) ~isempty(x.bt), S);
fprintf("Connected %d/%d sensors.\n", nnz(isOn), numel(S));

% --------- Plot (ax_g for each sensor) ----------
N = 400;
fig = figure('Name','5-sensor realtime (binary)','Color','w');
ax = axes(fig);
hold(ax,'on'); grid(ax,'on');
xlabel(ax,'Time (s)'); ylabel(ax,'ax (g)');
title(ax,'Realtime ax (binary decode)');

for i = 1:numel(S)
    S(i).tbuf  = nan(N,1);
    S(i).axbuf = nan(N,1);
    S(i).h = plot(ax, S(i).tbuf, S(i).axbuf, '-');
end
legend(ax, string({S.tag}), 'Interpreter','none', 'Location','best');
hold(ax,'off');

tStart = tic;
tLastPrint = tic;

try
    while ishandle(fig)

        % ---- Read bytes from each sensor (all available) ----
        for i = 1:numel(S)
            if isempty(S(i).bt), continue; end

            n = S(i).bt.NumBytesAvailable;
            if n > 0
                bytes = read(S(i).bt, n, "uint8");
                S(i).totalBytes = S(i).totalBytes + n;
                S(i).buf = [S(i).buf; bytes(:)];
            end
        end

        % ---- Decode ALL complete packets from each sensor buffer ----
        for i = 1:numel(S)
            if isempty(S(i).bt), continue; end

            [S(i), didPlot] = decodeAllPackets(S(i), PKT_LEN, SYNC1, SYNC2, ACCEL_SCALE, GYRO_SCALE);

            % update plot if we decoded anything
            if didPlot
                set(S(i).h, 'XData', S(i).tbuf, 'YData', S(i).axbuf);
            end
        end

        drawnow limitrate;

        % ---- Print stats every 1 second ----
        if toc(tLastPrint) >= 1
            elapsed = toc(tStart);
            fprintf("[%.1fs]\n", elapsed);
            for i = 1:numel(S)
                if isempty(S(i).bt)
                    fprintf("  %-10s OFFLINE\n", S(i).tag);
                else
                    fprintf("  %-10s bytes=%d pkts=%d buf=%d NumBytesAvail=%d\n", ...
                        S(i).tag, S(i).totalBytes, S(i).pktCount, numel(S(i).buf), S(i).bt.NumBytesAvailable);
                end
            end
            tLastPrint = tic;
        end

        % small pause reduces CPU but keeps up fine
        pause(0.001);
    end

catch ME
    fprintf("Stopped due to error: %s\n", ME.message);
end

% --------- Cleanup connections ----------
for i = 1:numel(S)
    try, clear S(i).bt; catch, end
end

% --------- Save CSVs ----------
% outDir = "Data";
% if ~exist(outDir,'dir'), mkdir(outDir); end
% 
% stamp = datetime("now");
% 
% M = [S(1).dataMat, S(2).dataMat, S(3).dataMat, S(4).dataMat, S(5).dataMat];
% varNames = { ...
%     'Head seq', 'Head Time', 'Head Ax','Head Ay','Head Az','Head Gx','Head Gy','Head Gz', ...
%     'Right Hand seq', 'Right Hand Time', 'Right Hand Ax','Right Hand Ay','Right Hand Az','Right Hand Gx','Right Hand Gy','Right Hand Gz', ...
%     'Left Hand seq', 'Left Hand Time', 'Left Hand Ax','Left Hand Ay','Left Hand Az','Left Hand Gx','Left Hand Gy','Left Hand Gz', ...
%     'Right Foot seq', 'Right Foot Time', 'Right Foot Ax','Right Foot Ay','Right Foot Az','Right Foot Gx','Right Foot Gy','Right Foot Gz', ...
%     'Left Foot seq', 'Left Foot Time', 'Left Foot Ax','Left Foot Ay','Left Foot Az','Left Foot Gx','Left Foot Gy','Left Foot Gz' };
% 
% T = array2table(M, 'VariableNames', varNames);
% fname = fullfile(outDir, sprintf('patient %s.xlsx', stamp));
% writetable(T, fname);
fprintf("[%s] Saved: %s (%d samples)\n", S(i).tag, fname, height(T));

outDir = "Data";
if ~exist(outDir,'dir'), mkdir(outDir); end

anySaved = false;
stamp = datestr(now,'yyyymmdd_HHMMSS');

for i = 1:numel(S)
    if isempty(S(i).dataMat)
        fprintf("[%s] No data collected, not saved.\n", S(i).tag);
        continue;
    end

    T = array2table(S(i).dataMat, 'VariableNames', ...
        {'seq','time_ms','ax_g','ay_g','az_g','gx_dps','gy_dps','gz_dps'});

    fname = fullfile(outDir, sprintf('%s_%s.csv', S(i).tag, stamp));
    writetable(T, fname);
    fprintf("[%s] Saved: %s (%d samples)\n", S(i).tag, fname, height(T));
    anySaved = true;
end

if ~anySaved
    disp("No sensor produced data, nothing saved.");
end

%% --------- helper: decode ALL packets currently available in buffer ---------
function [Si, didPlot] = decodeAllPackets(Si, PKT_LEN, SYNC1, SYNC2, ACCEL_SCALE, GYRO_SCALE)
didPlot = false;

buf = Si.buf;

while numel(buf) >= 2
    idx = find(buf(1:end-1)==SYNC1 & buf(2:end)==SYNC2, 1, 'first');
    if isempty(idx)
        % keep last byte only (could be SYNC1)
        buf = buf(end);
        break;
    end

    if idx > 1
        buf = buf(idx:end);
    end

    if numel(buf) < PKT_LEN
        break; % wait for more bytes
    end

    pkt = buf(1:PKT_LEN);
    buf = buf(PKT_LEN+1:end);

    seq  = typecast(uint8(pkt(3:4)), "uint16");
    t_ms = typecast(uint8(pkt(5:8)), "uint32");
    ax_c = typecast(uint8(pkt(9:10)),  "int16");
    ay_c = typecast(uint8(pkt(11:12)), "int16");
    az_c = typecast(uint8(pkt(13:14)), "int16");
    gx_c = typecast(uint8(pkt(15:16)), "int16");
    gy_c = typecast(uint8(pkt(17:18)), "int16");
    gz_c = typecast(uint8(pkt(19:20)), "int16");

    % loss detection (per sensor)
    if ~isempty(Si.lastSeq)
        missed = double(seq) - double(Si.lastSeq) - 1;
        if missed > 0
            fprintf("MISSED %d packets [%s] (seq %d -> %d)\n", missed, Si.tag, Si.lastSeq, seq);
        end
    end
    Si.lastSeq = seq;

    % scale
    t_s  = double(t_ms) / 1000.0;
    ax_g = double(ax_c) / ACCEL_SCALE;
    ay_g = double(ay_c) / ACCEL_SCALE;
    az_g = double(az_c) / ACCEL_SCALE;
    gx_d = double(gx_c) / GYRO_SCALE;
    gy_d = double(gy_c) / GYRO_SCALE;
    gz_d = double(gz_c) / GYRO_SCALE;

    Si.dataMat(end+1,:) = [double(seq), double(t_ms), ax_g, ay_g, az_g, gx_d, gy_d, gz_d];
    Si.pktCount = Si.pktCount + 1;

    % plot ax_g (ring buffer)
    Si.tbuf  = [Si.tbuf(2:end);  t_s];
    Si.axbuf = [Si.axbuf(2:end); ax_g];
    didPlot = true;
end

Si.buf = buf;
end
