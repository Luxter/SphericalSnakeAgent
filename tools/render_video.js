/**
 * tools/render_video.js
 *
 * Renders a GIF from an agent trace produced by tools/agent_trace.py.
 * Loads game/snake.js UNMODIFIED via vm.runInThisContext — identical to
 * snake_trace.js — but swaps the no-op canvas Proxy for a real node-canvas
 * so that snake.js's own render() function draws actual pixels.
 *
 * Math.random is patched to pop recorded pellet positions from the trace,
 * guaranteeing exact reproduction of the episode the agent experienced.
 *
 * Usage:
 *   node tools/render_video.js --trace agent_trace.json --output agent.gif
 *   node tools/render_video.js --trace agent_trace.json --output agent.gif --episode 0 --frame-skip 4 --fps 20
 */

'use strict';

const vm      = require('vm');
const fs      = require('fs');
const path    = require('path');
const { createCanvas } = require('canvas');
const GIFEncoder        = require('gifencoder');

// ---------------------------------------------------------------------------
// 1. Parse CLI args
// ---------------------------------------------------------------------------
var _args        = process.argv.slice(2);
var _tracePath   = null;
var _outputPath  = 'agent.gif';
var _episodeIdx  = 0;
var _frameSkip   = 4;   // render every Nth tick
var _fps         = 20;

for (var _i = 0; _i < _args.length; _i++) {
    if (_args[_i] === '--trace')      _tracePath   = _args[++_i];
    if (_args[_i] === '--output')     _outputPath  = _args[++_i];
    if (_args[_i] === '--episode')    _episodeIdx  = parseInt(_args[++_i], 10);
    if (_args[_i] === '--frame-skip') _frameSkip   = parseInt(_args[++_i], 10);
    if (_args[_i] === '--fps')        _fps         = parseInt(_args[++_i], 10);
}

if (!_tracePath) {
    process.stderr.write('Usage: node render_video.js --trace <path> [--output agent.gif] [--episode 0] [--frame-skip 4] [--fps 20]\n');
    process.exit(1);
}

// ---------------------------------------------------------------------------
// 2. Load trace and select episode
// ---------------------------------------------------------------------------
var _trace   = JSON.parse(fs.readFileSync(_tracePath, 'utf8'));
var _episode = _trace.episodes[_episodeIdx];
if (!_episode) {
    process.stderr.write('Episode index ' + _episodeIdx + ' out of range (trace has ' + _trace.episodes.length + ' episodes)\n');
    process.exit(1);
}

console.log('Episode ' + _episodeIdx + ': score=' + _episode.score + ', ticks=' + _episode.actions.length + ', pellets_spawned=' + _episode.pellets.length);

// ---------------------------------------------------------------------------
// 3. Patch Math.random to replay recorded pellet positions.
//    regeneratePellet() calls Math.random() twice: once for theta, once for phi.
//    We intercept paired calls and return sin/cos-compatible values that
//    reconstruct the exact recorded [x, y, z] positions.
//
//    Strategy: for each recorded pellet [px, py, pz]:
//      phi   = acos(pz)                  → Math.random() call 2 returns phi/π
//      theta = atan2(py, px*sin(phi)...) → Math.random() call 1 returns theta/(2π)
//    Then pointFromSpherical(theta, phi) reproduces the exact point.
// ---------------------------------------------------------------------------
var _pelletQueue = _episode.pellets.slice();  // copy, will be shifted
var _randCallsThisRegenerateIteration = 0;
var _currentTheta = 0;
var _currentPhi   = 0;

function _prepareNextPellet() {
    if (_pelletQueue.length === 0) {
        // Fallback: should not happen if trace is complete
        _currentTheta = 0;
        _currentPhi   = Math.PI / 4;
        return;
    }
    var p = _pelletQueue.shift();
    var pz  = p[2];
    var phi = Math.acos(Math.max(-1, Math.min(1, pz)));
    var sinPhi = Math.sin(phi);
    var theta;
    if (sinPhi < 1e-9) {
        theta = 0;
    } else {
        theta = Math.atan2(p[1], p[0]);
        if (theta < 0) theta += 2 * Math.PI;
    }
    _currentTheta = theta;
    _currentPhi   = phi;
}

_prepareNextPellet();  // ready for the first call from init()
_randCallsThisRegenerateIteration = 0;

Math.random = function() {
    var result;
    if (_randCallsThisRegenerateIteration === 0) {
        result = _currentTheta / (2 * Math.PI);
    } else {
        result = _currentPhi / Math.PI;
        _prepareNextPellet();
    }
    _randCallsThisRegenerateIteration = (_randCallsThisRegenerateIteration + 1) % 2;
    return result;
};

// ---------------------------------------------------------------------------
// 4. Real node-canvas — replaces the no-op Proxy from snake_trace.js
// ---------------------------------------------------------------------------
var _canvas    = createCanvas(358, 360);
var _ctx       = _canvas.getContext('2d');

function _makeElement() {
    return {
        style: {},
        innerHTML: '',
        classList: { add: function(){}, remove: function(){} },
        addEventListener: function(){},
    };
}

global.document = {
    querySelector:        function()  { return _makeElement(); },
    getElementById:       function()  { return _makeElement(); },
    getElementsByTagName: function(t) {
        if (t === 'canvas') return [_canvas];
        return [_makeElement()];
    },
};

global.window = {
    addEventListener:      function() {},
    requestAnimationFrame: function() {},  // no-op — we drive ticks manually
    location: { reload: function() {} },
};

// ---------------------------------------------------------------------------
// 5. Load game/snake.js in THIS global context — identical to snake_trace.js.
//    init() runs immediately: builds sphere grid, snake, calls regeneratePellet()
//    (which triggers Math.random twice → consumes pellets[0]).
// ---------------------------------------------------------------------------
var _snakeJsPath = path.join(__dirname, '..', 'game', 'snake.js');
vm.runInThisContext(fs.readFileSync(_snakeJsPath, 'utf8'), { filename: _snakeJsPath });

// After this line: snake, pellet, direction, stopped, snakeVelocity,
// checkCollisions(), applySnakeRotation(), rotateZ(), rotateY(), render() are live.

// ---------------------------------------------------------------------------
// 6. GIF encoder setup
// ---------------------------------------------------------------------------
var _encoder = new GIFEncoder(358, 360);
var _outStream = fs.createWriteStream(_outputPath);
_encoder.createReadStream().pipe(_outStream);
_encoder.start();
_encoder.setRepeat(0);    // loop forever
_encoder.setDelay(Math.round(1000 / _fps));
_encoder.setQuality(10);

// ---------------------------------------------------------------------------
// 7. Drive ticks — mirrors snake_trace.js tick loop exactly, but calls
//    render() and captures frames at every _frameSkip-th tick.
// ---------------------------------------------------------------------------
var _actions = _episode.actions;
var _frameCount = 0;

for (var t = 0; t < _actions.length; t++) {
    // step 1: collision check
    checkCollisions();

    if (stopped) break;

    // step 2: action → direction
    var _action = _actions[t];
    if (_action === 1) direction -= 0.08;
    if (_action === 2) direction += 0.08;

    // step 3: move snake
    applySnakeRotation();

    // step 4: world rotation
    rotateZ(-direction);
    rotateY(-snakeVelocity);
    rotateZ(direction);

    // step 5: capture frame
    if (t % _frameSkip === 0) {
        render();
        _encoder.addFrame(_ctx);
        _frameCount++;
    }
}

// Capture final frame regardless of frame-skip
render();
_encoder.addFrame(_ctx);

_encoder.finish();
_outStream.on('finish', function() {
    console.log('GIF written: ' + _outputPath + '  (' + _frameCount + ' frames)');
});
