
"use strict";

let ObjectCount = require('./ObjectCount.js');
let BoundingBox = require('./BoundingBox.js');
let BoundingBoxes = require('./BoundingBoxes.js');
let CheckForObjectsFeedback = require('./CheckForObjectsFeedback.js');
let CheckForObjectsResult = require('./CheckForObjectsResult.js');
let CheckForObjectsActionResult = require('./CheckForObjectsActionResult.js');
let CheckForObjectsAction = require('./CheckForObjectsAction.js');
let CheckForObjectsActionFeedback = require('./CheckForObjectsActionFeedback.js');
let CheckForObjectsGoal = require('./CheckForObjectsGoal.js');
let CheckForObjectsActionGoal = require('./CheckForObjectsActionGoal.js');

module.exports = {
  ObjectCount: ObjectCount,
  BoundingBox: BoundingBox,
  BoundingBoxes: BoundingBoxes,
  CheckForObjectsFeedback: CheckForObjectsFeedback,
  CheckForObjectsResult: CheckForObjectsResult,
  CheckForObjectsActionResult: CheckForObjectsActionResult,
  CheckForObjectsAction: CheckForObjectsAction,
  CheckForObjectsActionFeedback: CheckForObjectsActionFeedback,
  CheckForObjectsGoal: CheckForObjectsGoal,
  CheckForObjectsActionGoal: CheckForObjectsActionGoal,
};
