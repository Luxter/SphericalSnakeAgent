/**
 * tools/snake_trace.js  (Option B)
 *
 * Loads docs/snake.js UNMODIFIED via vm.runInThisContext so its physics
 * functions and globals run in this process's global scope with zero copying.
 * Only changes from the browser environment:
 *   - Minimal DOM stubs satisfy snake.js's top-level querySelector/addEventListener calls
 *   - Math.random is replaced with a deterministic LCG before the script executes
 *   - window.requestAnimationFrame is a no-op (we drive ticks manually)
 *   - showEnd() still sets stopped=true; we read that flag instead of watching the DOM
 *
 * Usage:
 *   node tools/snake_trace.js --seed 42 --actions '[1,1,1,0,2,1]'
 *
 * Output: JSON array, one object per tick:
 *   { tick, direction, head, snake_length, pellet, posQueue_0 }
 */

'use strict';

const vm   = require('vm');
const fs   = require('fs');
const path = require('path');

// ---------------------------------------------------------------------------
// 1. Deterministic LCG — identical parameters to tools/snake_trace.py
// ---------------------------------------------------------------------------
var _lcgState = 0;

function _lcgSeed(s) { _lcgState = (s >>> 0); }

function _lcgRand() {
    _lcgState = ((Math.imul(_lcgState, 1664525) + 1013904223) >>> 0);
    return _lcgState / 4294967296;
}

// ---------------------------------------------------------------------------
// 2. Minimal DOM stubs — just enough for snake.js to load and run
// ---------------------------------------------------------------------------
function _makeElement() {
    var el = {
        style: {},
        innerHTML: '',
        classList: { add: function(){}, remove: function(){} },
        addEventListener: function(){},
    };
    return el;
}

var _canvasCtx = new Proxy({}, { get: function() { return function(){}; } });
var _canvasEl  = { getContext: function() { return _canvasCtx; }, width: 358, height: 360 };

global.document = {
    querySelector:        function()  { return _makeElement(); },
    getElementById:       function()  { return _makeElement(); },
    getElementsByTagName: function(t) {
        if (t === 'canvas') return [_canvasEl];
        return [_makeElement()];
    },
};

global.window = {
    addEventListener:     function() {},
    requestAnimationFrame:function() {},   // no-op — we drive ticks manually
    location: { reload: function() {} },
};

// ---------------------------------------------------------------------------
// 3. Seed LCG and replace Math.random BEFORE loading snake.js,
//    so init() → regeneratePellet() uses it from the very first call.
//    Seed is parsed from CLI args here, snake.js is loaded just below.
// ---------------------------------------------------------------------------
var _args    = process.argv.slice(2);
var _seed    = 42;
var _actions = [];

for (var _i = 0; _i < _args.length; _i++) {
    if (_args[_i] === '--seed')    _seed    = parseInt(_args[++_i], 10);
    if (_args[_i] === '--actions') _actions = JSON.parse(_args[++_i]);
}

_lcgSeed(_seed);
Math.random = _lcgRand;   // patch before load so init() → regeneratePellet() uses LCG

// ---------------------------------------------------------------------------
// 4. Load docs/snake.js in THIS global context.
//    vm.runInThisContext makes all its `var` globals (snake, pellet, direction,
//    stopped, snakeVelocity, …) and functions (rotateZ, rotateY,
//    applySnakeRotation, checkCollisions, …) accessible directly below.
//    init() at the bottom of snake.js runs here, building the initial state.
// ---------------------------------------------------------------------------
var _snakeJsPath = path.join(__dirname, '..', 'docs', 'snake.js');
vm.runInThisContext(fs.readFileSync(_snakeJsPath, 'utf8'), { filename: _snakeJsPath });

// After the above line, all of snake.js's globals are live in this scope:
//   snake, pellet, direction, stopped, snakeVelocity,
//   checkCollisions(), applySnakeRotation(), rotateZ(), rotateY(), etc.

// ---------------------------------------------------------------------------
// 5. Snapshot helper
// ---------------------------------------------------------------------------
function _snapshot(tick) {
    var pq0 = snake[0].posQueue.map(function(p) {
        return p ? { x: p.x, y: p.y, z: p.z } : null;
    });
    return {
        tick:         tick,
        direction:    direction,
        head:         { x: snake[0].x, y: snake[0].y, z: snake[0].z },
        snake_length: snake.length,
        pellet:       { x: pellet.x,   y: pellet.y,   z: pellet.z   },
        posQueue_0:   pq0,
    };
}

// ---------------------------------------------------------------------------
// 6. Drive ticks manually — mirrors the JS update() while-loop body exactly
// ---------------------------------------------------------------------------
(function main() {
    var trace = [];

    for (var t = 0; t < _actions.length; t++) {
        // step 1: collision check
        checkCollisions();

        if (stopped) {
            trace.push(Object.assign(_snapshot(t), { event: 'terminated' }));
            break;
        }

        // Was a pellet eaten this tick? checkCollisions() already mutated
        // snake_length and called regeneratePellet() if so — detect via
        // comparing length against previous tick.
        var prevLen = t === 0 ? 8 : trace[t - 1].snake_length;
        var eaten   = (snake.length > prevLen);

        // step 2: action → direction
        var action = _actions[t];
        if (action === 1) direction -= 0.08;
        if (action === 2) direction += 0.08;

        // step 3: move snake
        applySnakeRotation();

        // step 4: world rotation (rotateZ/rotateY with no pt arg use allPoints()
        //         internally, which includes grid points + pellet + snake + queues)
        rotateZ(-direction);
        rotateY(-snakeVelocity);
        rotateZ(direction);

        var snap = _snapshot(t);
        if (eaten) snap.event = 'pellet_eaten';
        trace.push(snap);
    }

    process.stdout.write(JSON.stringify(trace, null, 2) + '\n');
})();
