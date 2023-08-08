
"use strict";

let Led = require('./Led.js');
let SensorState = require('./SensorState.js');
let CliffEvent = require('./CliffEvent.js');
let ControllerInfo = require('./ControllerInfo.js');
let Sound = require('./Sound.js');
let PowerSystemEvent = require('./PowerSystemEvent.js');
let ScanAngle = require('./ScanAngle.js');
let DigitalInputEvent = require('./DigitalInputEvent.js');
let MotorPower = require('./MotorPower.js');
let ExternalPower = require('./ExternalPower.js');
let BumperEvent = require('./BumperEvent.js');
let VersionInfo = require('./VersionInfo.js');
let KeyboardInput = require('./KeyboardInput.js');
let DockInfraRed = require('./DockInfraRed.js');
let RobotStateEvent = require('./RobotStateEvent.js');
let ButtonEvent = require('./ButtonEvent.js');
let WheelDropEvent = require('./WheelDropEvent.js');
let DigitalOutput = require('./DigitalOutput.js');
let AutoDockingResult = require('./AutoDockingResult.js');
let AutoDockingActionGoal = require('./AutoDockingActionGoal.js');
let AutoDockingActionFeedback = require('./AutoDockingActionFeedback.js');
let AutoDockingAction = require('./AutoDockingAction.js');
let AutoDockingGoal = require('./AutoDockingGoal.js');
let AutoDockingActionResult = require('./AutoDockingActionResult.js');
let AutoDockingFeedback = require('./AutoDockingFeedback.js');

module.exports = {
  Led: Led,
  SensorState: SensorState,
  CliffEvent: CliffEvent,
  ControllerInfo: ControllerInfo,
  Sound: Sound,
  PowerSystemEvent: PowerSystemEvent,
  ScanAngle: ScanAngle,
  DigitalInputEvent: DigitalInputEvent,
  MotorPower: MotorPower,
  ExternalPower: ExternalPower,
  BumperEvent: BumperEvent,
  VersionInfo: VersionInfo,
  KeyboardInput: KeyboardInput,
  DockInfraRed: DockInfraRed,
  RobotStateEvent: RobotStateEvent,
  ButtonEvent: ButtonEvent,
  WheelDropEvent: WheelDropEvent,
  DigitalOutput: DigitalOutput,
  AutoDockingResult: AutoDockingResult,
  AutoDockingActionGoal: AutoDockingActionGoal,
  AutoDockingActionFeedback: AutoDockingActionFeedback,
  AutoDockingAction: AutoDockingAction,
  AutoDockingGoal: AutoDockingGoal,
  AutoDockingActionResult: AutoDockingActionResult,
  AutoDockingFeedback: AutoDockingFeedback,
};
