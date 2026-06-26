function DataCollector_Timer()
% ===================== MAIN: constants + actions ======================
CFG.devices.Head       = "TTGO-06";
CFG.devices.Right_Hand = "TTGO-11";
CFG.devices.Left_Hand  = "TTGO-12";
CFG.devices.Right_Foot = "TTGO-18";
CFG.devices.Left_Foot  = "TTGO-19";
CFG.btChannel = 1;

% timer 
CFG.tickPeriod   = 0.011;    % 100 Hz nominal timer tick

% protocol
CFG.acquireDur_s = 510;       % record 5 seconds per segment
CFG.keepDur_s = 499;       % keep last 5 sec
CFG.cooldown_s   = 4;       % wait 3 seconds after saving
CFG.maxSegments  = 30;      % 30 files in Start protocol

% ------------- UI -------------
UI = buildFigure();

% ------------- State -------------
STATE.isRunning   = false;
STATE.mode        = 'idle';   % 'idle' | 'acquire' | 'cooldown' | 'done' | 'single_acquire' | 'single_done'
STATE.segIdx      = 0;
STATE.tStart      = [];       % tic for acquire/cooldown window
STATE.tmr         = [];

% segment buffers (reset every acquire window)
clearSegBuffers();

% live sample counter (for display)
STATE.liveCount = 0;

% callbacks
set(UI.btnStart,     'Callback', @(~,~) onStart());
set(UI.btnSingle,    'Callback', @(~,~) onSingle());
set(UI.btnStop,      'Callback', @(~,~) onStop());
set(UI.btnReconnect, 'Callback', @(~,~) onReconnect());
set(UI.fig, 'DeleteFcn', @(~,~) onClose());

% ------------- loop -------------
SENS.Head = [];
SENS.Right_Hand = [];
SENS.Left_Hand = [];
SENS.Right_Foot = [];
SENS.Left_Foot = [];

if isempty(STATE.tmr) || ~isvalid(STATE.tmr)
    STATE.tmr = timer('ExecutionMode','fixedRate', ...
                      'Period', CFG.tickPeriod, ...
                      'TimerFcn', @tickLogic);
end
clearSegBuffers();
start(STATE.tmr);

% ------------- Connection on load -------------
SENS.Head       = btOpen(CFG.devices.Head,       CFG.btChannel, UI);
connectSensors(UI, SENS);
SENS.Right_Hand = btOpen(CFG.devices.Right_Hand, CFG.btChannel, UI);
connectSensors(UI, SENS);
SENS.Left_Hand  = btOpen(CFG.devices.Left_Hand,  CFG.btChannel, UI);
connectSensors(UI, SENS);
SENS.Right_Foot = btOpen(CFG.devices.Right_Foot, CFG.btChannel, UI);
connectSensors(UI, SENS);
SENS.Left_Foot  = btOpen(CFG.devices.Left_Foot,  CFG.btChannel, UI);
connectSensors(UI, SENS);

% ===================== callbacks ======================
    function onStart()

        if STATE.isRunning, return; end
        % get base name
        base = get(UI.editFile,'String');
        if isempty(base)
            base = ['session_', datetime("now")];
        end
        % ensure Data folder
        if ~exist('Data','dir'), mkdir('Data'); end
        set(UI.editFile,'String', base);

        % reset protocol state
        STATE.isRunning = true;
        STATE.mode      = 'acquire';
        STATE.segIdx    = 1;
        STATE.liveCount = 0;
        STATE.tStart = tic;

        set(UI.btnStart,'Enable','off');
        set(UI.btnSingle,'Enable','off');
        set(UI.btnStop, 'Enable','on');
        set(UI.txtMsg, 'String', sprintf('Segment %02d/%02d: acquiring %g s...', STATE.segIdx, CFG.maxSegments, CFG.acquireDur_s));
        set(UI.txtCount,'String','Samples: 0');

        clearSegBuffers();
    end

    function onSingle()
        if STATE.isRunning
            set(UI.txtMsg,'String','Busy: stop or wait until current segment ends.');
            return;
        end
        % ensure base name & Data folder
        base = get(UI.editFile,'String');
        if isempty(base)
            base = ['session_', datetime("now")];
            set(UI.editFile,'String', base);
        end
        if ~exist('Data','dir'), mkdir('Data'); end

        % prep one-shot
        STATE.isRunning = true;
        STATE.mode      = 'single_acquire';
        STATE.liveCount = 0;
        clearSegBuffers();
        set(UI.txtCount,'String','Samples: 0');
        set(UI.txtMsg,'String',sprintf('Single Take: acquiring %g s ...', CFG.acquireDur_s));
        set(UI.btnStart,'Enable','off');
        set(UI.btnSingle,'Enable','off');
        set(UI.btnStop, 'Enable','on');

        STATE.tStart = tic;
        clearSegBuffers();
    end

    function onStop()
        if ~isempty(STATE.tmr) && isvalid(STATE.tmr), stop(STATE.tmr); end
        STATE.isRunning = false;
        STATE.mode = 'idle';
        set(UI.btnStart,'Enable','on');
        set(UI.btnSingle,'Enable','on');
        set(UI.btnStop, 'Enable','off');
        set(UI.txtMsg,'String','Stopped.');
    end

    function onReconnect()
        set(UI.txtMsg,'String','Reconnecting...');
        closeSensors(SENS);
        pause(0.3);
        SENS = connectSensors(CFG, UI);
        % UI state preserved
    end

    function onClose()
        if ~isempty(STATE.tmr) && isvalid(STATE.tmr)
            try stop(STATE.tmr); catch, end
            delete(STATE.tmr);
        end
        closeSensors(SENS);
        delete(UI.fig);
    end

% ===================== timer tick (nested, no args) ======================
    function tickLogic(~, ~)   
        % read all sensors (non-blocking)
        [okH,  tH, aH,  gH ] = readOne(SENS.Head);
        [okRH, tRH, aRH, gRH] = readOne(SENS.Right_Hand);
        [okLH, tLH, aLH, gLH] = readOne(SENS.Left_Hand);
        [okRF, tRF, aRF, gRF] = readOne(SENS.Right_Foot);
        [okLF, tLF, aLF, gLF] = readOne(SENS.Left_Foot);

        ok = false;
        if (okH && okRH && okLH && okRF && okLF) 
            ok = true;
        end

        % ========== SINGLE TAKE ==========
        if strcmpi(STATE.mode,'single_acquire')
            if (ok) 
                STATE.seg.t.Head = [STATE.seg.t.Head; tH];  STATE.seg.acc.Head = [STATE.seg.acc.Head; aH];  STATE.seg.gyro.Head = [STATE.seg.gyro.Head; gH];
                STATE.seg.t.RH   = [STATE.seg.t.RH;   tRH]; STATE.seg.acc.RH   = [STATE.seg.acc.RH;   aRH]; STATE.seg.gyro.RH   = [STATE.seg.gyro.RH;   gRH];
                STATE.seg.t.LH   = [STATE.seg.t.LH;   tLH]; STATE.seg.acc.LH   = [STATE.seg.acc.LH;   aLH]; STATE.seg.gyro.LH   = [STATE.seg.gyro.LH;   gLH];
                STATE.seg.t.RF   = [STATE.seg.t.RF;   tRF]; STATE.seg.acc.RF   = [STATE.seg.acc.RF;   aRF]; STATE.seg.gyro.RF   = [STATE.seg.gyro.RF;   gRF];
                STATE.seg.t.LF   = [STATE.seg.t.LF;   tLF]; STATE.seg.acc.LF   = [STATE.seg.acc.LF;   aLF]; STATE.seg.gyro.LF   = [STATE.seg.gyro.LF;   gLF];
            end

            STATE.liveCount = min([ size(STATE.seg.acc.Head,1), size(STATE.seg.acc.RH,1), size(STATE.seg.acc.LH,1), ...
                                    size(STATE.seg.acc.RF,1),   size(STATE.seg.acc.LF,1) ]);
            set(UI.txtCount,'String',sprintf('Samples: %d', STATE.liveCount));

            if STATE.liveCount >= CFG.acquireDur_s
                base = get(UI.editFile,'String');
                oneName = fullfile('Data', sprintf('%s.single_%s.xlsx', base, datetime("now")));
                try
                    saveOneSegment(oneName, STATE.seg, CFG.keepDur_s);
                    set(UI.txtMsg,'String', ['Single Take saved: ', oneName]);
                catch ME
                    set(UI.txtMsg,'String', ['Single Take save error: ', ME.message]);
                end
                % finish single take
                STATE.mode = 'single_done';
            end

        elseif strcmpi(STATE.mode,'single_done')
            % stop timer & reset UI to idle
            try stop(STATE.tmr); catch, end
            STATE.isRunning = false;
            set(UI.btnStart,'Enable','on');
            set(UI.btnSingle,'Enable','on');
            set(UI.btnStop, 'Enable','off');
            STATE.mode = 'idle';

        % ========== MULTI-CYCLE: ACQUIRE ==========
        elseif strcmpi(STATE.mode,'acquire')
            if (ok) 
                STATE.seg.t.Head = [STATE.seg.t.Head; tH];  STATE.seg.acc.Head = [STATE.seg.acc.Head; aH];  STATE.seg.gyro.Head = [STATE.seg.gyro.Head; gH];
                STATE.seg.t.RH   = [STATE.seg.t.RH;   tRH]; STATE.seg.acc.RH   = [STATE.seg.acc.RH;   aRH]; STATE.seg.gyro.RH   = [STATE.seg.gyro.RH;   gRH];
                STATE.seg.t.LH   = [STATE.seg.t.LH;   tLH]; STATE.seg.acc.LH   = [STATE.seg.acc.LH;   aLH]; STATE.seg.gyro.LH   = [STATE.seg.gyro.LH;   gLH];
                STATE.seg.t.RF   = [STATE.seg.t.RF;   tRF]; STATE.seg.acc.RF   = [STATE.seg.acc.RF;   aRF]; STATE.seg.gyro.RF   = [STATE.seg.gyro.RF;   gRF];
                STATE.seg.t.LF   = [STATE.seg.t.LF;   tLF]; STATE.seg.acc.LF   = [STATE.seg.acc.LF;   aLF]; STATE.seg.gyro.LF   = [STATE.seg.gyro.LF;   gLF];
            end
            
            STATE.liveCount = min([ size(STATE.seg.acc.Head,1), size(STATE.seg.acc.RH,1), size(STATE.seg.acc.LH,1), ...
                                    size(STATE.seg.acc.RF,1),   size(STATE.seg.acc.LF,1) ]);
            set(UI.txtCount,'String',sprintf('Samples: %d', STATE.liveCount));

            if STATE.liveCount >= CFG.acquireDur_s
                % save current segment as file
                base = get(UI.editFile,'String');
                outName = fullfile('Data', sprintf('%s.%02d.xlsx', base, STATE.segIdx));
                try
                    saveOneSegment(outName, STATE.seg, CFG.keepDur_s);
                    set(UI.txtMsg,'String', sprintf('Saved: %s  |  Cooldown %g s...', sprintf('%s.%02d.xlsx', base, STATE.segIdx), CFG.cooldown_s));
                catch ME
                    set(UI.txtMsg,'String', ['Save error: ', ME.message]);
                end

                % move to cooldown
                STATE.mode   = 'cooldown';
                STATE.tStart = tic; % reuse timer for cooldown
            end

        % ========== MULTI-CYCLE: COOLDOWN ==========
        elseif strcmpi(STATE.mode,'cooldown')
            % wait only; no appending
            if toc(STATE.tStart) >= CFG.cooldown_s
                % next segment or finish
                STATE.segIdx = STATE.segIdx + 1;
                if STATE.segIdx > CFG.maxSegments
                    STATE.mode = 'done';
                    set(UI.txtMsg,'String','All segments complete. You can Close or Start again.');
                    set(UI.btnStart,'Enable','on');
                    set(UI.btnSingle,'Enable','on');
                    set(UI.btnStop, 'Enable','off');
                    try stop(STATE.tmr); catch, end
                    STATE.isRunning = false;
                else
                    clearSegBuffers();
                    if ~isempty(SENS.Head), flush(SENS.Head); end
                    if ~isempty(SENS.Right_Hand), flush(SENS.Right_Hand); end
                    if ~isempty(SENS.Left_Hand), flush(SENS.Left_Hand); end
                    if ~isempty(SENS.Right_Foot), flush(SENS.Right_Foot); end
                    if ~isempty(SENS.Left_Foot), flush(SENS.Left_Foot); end
                    STATE.liveCount = 0;
                    set(UI.txtCount,'String','Samples: 0');
                    set(UI.txtMsg,'String', sprintf('Segment %02d/%02d: acquiring %g s...', STATE.segIdx, CFG.maxSegments, CFG.acquireDur_s));
                    STATE.mode   = 'acquire';
                    STATE.tStart = tic;
                end
            end
        else
            clearSegBuffers();
            if ~isempty(SENS.Head), flush(SENS.Head); end
            if ~isempty(SENS.Right_Hand), flush(SENS.Right_Hand); end
            if ~isempty(SENS.Left_Hand), flush(SENS.Left_Hand); end
            if ~isempty(SENS.Right_Foot), flush(SENS.Right_Foot); end
            if ~isempty(SENS.Left_Foot), flush(SENS.Left_Foot); end
        end

        drawnow limitrate
    end

% ===================== helpers ======================
    function clearSegBuffers()
        STATE.seg.t.Head = []; STATE.seg.acc.Head = []; STATE.seg.gyro.Head = [];
        STATE.seg.t.RH   = []; STATE.seg.acc.RH   = []; STATE.seg.gyro.RH   = [];
        STATE.seg.t.LH   = []; STATE.seg.acc.LH   = []; STATE.seg.gyro.LH   = [];
        STATE.seg.t.RF   = []; STATE.seg.acc.RF   = []; STATE.seg.gyro.RF   = [];
        STATE.seg.t.LF   = []; STATE.seg.acc.LF   = []; STATE.seg.gyro.LF   = [];
    end
end

% ===================== 1) CONNECTION =====================================
function connectSensors(UI, SENS)

setDot(UI.dot.Head, btIsOpen(SENS.Head));
setDot(UI.dot.RH,   btIsOpen(SENS.Right_Hand));
setDot(UI.dot.LH,   btIsOpen(SENS.Left_Hand));
setDot(UI.dot.RF,   btIsOpen(SENS.Right_Foot));
setDot(UI.dot.LF,   btIsOpen(SENS.Left_Foot));

if all([btIsOpen(SENS.Head), btIsOpen(SENS.Right_Hand), btIsOpen(SENS.Left_Hand), btIsOpen(SENS.Right_Foot), btIsOpen(SENS.Left_Foot)])
    set(UI.txtMsg,'String','All sensors connected.');
else
    set(UI.txtMsg,'String','Some sensors not connected (you can still run).');
end
end


% btOpen
function s = btOpen(name, channel, UI)
try
    s = bluetooth(name, channel);
    configureTerminator(s,"LF");
    flush(s);
catch ME
    set(UI.txtMsg,'String',sprintf('BT connect failed: %s (ch %d): %s',name,channel,ME.message));
    ME.message
    s = [];
end
end

% btIsOpen
function ok = btIsOpen(s)
ok = ~isempty(s);
if ok
    try nb = s.NumBytesAvailable; %#ok<NASGU>
    catch, ok = false; end
end
end

function closeSensors(SENS)
fn = fieldnames(SENS);
for i=1:numel(fn)
    so = SENS.(fn{i});
    if ~isempty(so)
        try delete(so); catch, end
    end
end
end

% ===================== 2) FIGURE / UI ====================================
function UI = buildFigure()
UI.fig = figure('Name','Five Sensor Collector','NumberTitle','off', ...
                'Position',[200,200,760,420]);

uicontrol('Style','text','String','Base file name:', ...
          'Position',[20,370,170,20],'HorizontalAlignment','left');
UI.editFile = uicontrol('Style','edit','Position',[20,345,300,25], ...
                        'HorizontalAlignment','left','String','patient');

UI.btnStart = uicontrol('Style','pushbutton','String','Start (30× 5s)', ...
                        'Position',[340,342,130,30]);
UI.btnSingle = uicontrol('Style','pushbutton','String','Single Take', ...
                        'Position',[480,342,110,30]);
UI.btnStop  = uicontrol('Style','pushbutton','String','Stop', ...
                        'Position',[600,342,110,30], 'Enable','off');
UI.btnReconnect = uicontrol('Style','pushbutton','String','Reconnect', ...
                            'Position',[600,310,110,26]);

% ---------- status dots ----------
y0=250; dy=28; x1=20; x2=160;
[~, UI.dot.Head] = statusRow('Head',       x1,y0,     x2,y0);
[~, UI.dot.RH]   = statusRow('Right Hand', x1,y0-dy,  x2,y0-dy);
[~, UI.dot.LH]   = statusRow('Left Hand',  x1,y0-2*dy,x2,y0-2*dy);
[~, UI.dot.RF]   = statusRow('Right Foot', x1,y0-3*dy,x2,y0-3*dy);
[~, UI.dot.LF]   = statusRow('Left Foot',  x1,y0-4*dy,x2,y0-4*dy);

UI.txtCount = uicontrol('Style','text','String','Samples: 0', ...
                        'Position',[340,250,300,20],'HorizontalAlignment','left');
UI.txtMsg   = uicontrol('Style','text','String','', ...
                        'Position',[340,220,360,20],'HorizontalAlignment','left');
end

function [hLabel,hDot] = statusRow(label, x1,y1, x2,y2)
hLabel = uicontrol('Style','text','String',label, ...
         'Position',[x1,y1,120,20],'HorizontalAlignment','left');
hDot   = uicontrol('Style','text','String','●', ...
         'Position',[x2,y2,20,20],'HorizontalAlignment','left','ForegroundColor',[0.7 0 0]);
end

function setDot(hDot,isOn)
if isOn, set(hDot,'ForegroundColor',[0 0.6 0]); else, set(hDot,'ForegroundColor',[0.7 0 0]); end
end

% ===================== 3) READ ONE LINE ==================================
function [ok, t, acc3, gyr3] = readOne(so)
ok=false; t=0; acc3=[NaN NaN NaN]; gyr3=[NaN NaN NaN];
if isempty(so), return; end
try
    if so.NumBytesAvailable > 0
        line = readline(so);
        p = sscanf(line,'%lu,%f,%f,%f,%f,%f,%f');
        if numel(p)==7
            t = p(1);
            acc3 = [p(2),p(3),p(4)];
            gyr3 = [p(5),p(6),p(7)];
            ok = true;
        end
    end
catch
    ok=false;
end
end

% ===================== 4) SAVE ONE SEGMENT ===============================
function saveOneSegment(filePath, SEG, keep)
% copy (so we can modify)
t = SEG.t; ACC = SEG.acc; GYR = SEG.gyro;

t.Head = t.Head(size(t.Head, 1) - keep : end, :);
t.RH = t.RH(size(t.RH, 1) - keep : end, :);
t.LH = t.LH(size(t.LH, 1) - keep : end, :);
t.RF = t.RF(size(t.RF, 1) - keep : end, :);
t.LF = t.LF(size(t.LF, 1) - keep : end, :);

ACC.Head = ACC.Head(size(ACC.Head, 1) - keep : end, :);
ACC.RH = ACC.RH(size(ACC.RH, 1) - keep : end, :);
ACC.LH = ACC.LH(size(ACC.LH, 1) - keep : end, :);
ACC.RF = ACC.RF(size(ACC.RF, 1) - keep : end, :);
ACC.LF = ACC.LF(size(ACC.LF, 1) - keep : end, :);

GYR.Head = GYR.Head(size(GYR.Head, 1) - keep : end, :);
GYR.RH = GYR.RH(size(GYR.RH, 1) - keep : end, :);
GYR.LH = GYR.LH(size(GYR.LH, 1) - keep : end, :);
GYR.RF = GYR.RF(size(GYR.RF, 1) - keep : end, :);
GYR.LF = GYR.LF(size(GYR.LF, 1) - keep : end, :);

% pad to same length (so columns fixed)
lens = [ size(ACC.Head,1), size(GYR.Head,1), ...
         size(ACC.RH,1),   size(GYR.RH,1), ...
         size(ACC.LH,1),   size(GYR.LH,1), ...
         size(ACC.RF,1),   size(GYR.RF,1), ...
         size(ACC.LF,1),   size(GYR.LF,1) ];
maxLen = max([lens,0]);
if maxLen==0, error('No samples captured in this window.'); end

pad3 = @(X) padStream3(X, maxLen);
ACC.Head=pad3(ACC.Head); GYR.Head=pad3(GYR.Head);
ACC.RH  =pad3(ACC.RH);   GYR.RH  =pad3(GYR.RH);
ACC.LH  =pad3(ACC.LH);   GYR.LH  =pad3(GYR.LH);
ACC.RF  =pad3(ACC.RF);   GYR.RF  =pad3(GYR.RF);
ACC.LF  =pad3(ACC.LF);   GYR.LF  =pad3(GYR.LF);

M = [ t.Head, ACC.Head, GYR.Head, ...
      t.RH,   ACC.RH,   GYR.RH,   ...
      t.LH,   ACC.LH,   GYR.LH,   ...
      t.RF,   ACC.RF,   GYR.RF,   ...
      t.LF,   ACC.LF,   GYR.LF ];

varNames = { ...
    'Head Time', 'Head Ax','Head Ay','Head Az','Head Gx','Head Gy','Head Gz', ...
    'Right Hand Time', 'Right Hand Ax','Right Hand Ay','Right Hand Az','Right Hand Gx','Right Hand Gy','Right Hand Gz', ...
    'Left Hand Time', 'Left Hand Ax','Left Hand Ay','Left Hand Az','Left Hand Gx','Left Hand Gy','Left Hand Gz', ...
    'Right Foot Time', 'Right Foot Ax','Right Foot Ay','Right Foot Az','Right Foot Gx','Right Foot Gy','Right Foot Gz', ...
    'Left Foot Time', 'Left Foot Ax','Left Foot Ay','Left Foot Az','Left Foot Gx','Left Foot Gy','Left Foot Gz' };

T = array2table(M, 'VariableNames', varNames);

[folder,~,~] = fileparts(filePath);
if ~isempty(folder) && ~exist(folder,'dir'), mkdir(folder); end
writetable(T, filePath);
end

function Xpad = padStream3(X, maxLen)
if isempty(X), Xpad = nan(maxLen,3); return; end
if size(X,2)<3, X = [X, nan(size(X,1), 3-size(X,2))]; end
if size(X,2)>3, X = X(:,1:3); end
n=size(X,1);
if n<maxLen, Xpad=[X; nan(maxLen-n,3)];
else,        Xpad=X(1:maxLen,:);
end
end

