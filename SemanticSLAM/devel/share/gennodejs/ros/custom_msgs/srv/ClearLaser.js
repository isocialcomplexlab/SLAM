// Auto-generated. Do not edit!

// (in-package custom_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let sensor_msgs = _finder('sensor_msgs');

//-----------------------------------------------------------


//-----------------------------------------------------------

class ClearLaserRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.weight = null;
      this.laser = null;
    }
    else {
      if (initObj.hasOwnProperty('weight')) {
        this.weight = initObj.weight
      }
      else {
        this.weight = 0;
      }
      if (initObj.hasOwnProperty('laser')) {
        this.laser = initObj.laser
      }
      else {
        this.laser = new sensor_msgs.msg.LaserScan();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ClearLaserRequest
    // Serialize message field [weight]
    bufferOffset = _serializer.int8(obj.weight, buffer, bufferOffset);
    // Serialize message field [laser]
    bufferOffset = sensor_msgs.msg.LaserScan.serialize(obj.laser, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ClearLaserRequest
    let len;
    let data = new ClearLaserRequest(null);
    // Deserialize message field [weight]
    data.weight = _deserializer.int8(buffer, bufferOffset);
    // Deserialize message field [laser]
    data.laser = sensor_msgs.msg.LaserScan.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += sensor_msgs.msg.LaserScan.getMessageSize(object.laser);
    return length + 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'custom_msgs/ClearLaserRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '00cb7b8b21b02f60fc59fa03cb0dfb8d';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # svr for the ClearCells service
    int8 weight
    sensor_msgs/LaserScan laser
    
    ================================================================================
    MSG: sensor_msgs/LaserScan
    # Single scan from a planar laser range-finder
    #
    # If you have another ranging device with different behavior (e.g. a sonar
    # array), please find or create a different message, since applications
    # will make fairly laser-specific assumptions about this data
    
    Header header            # timestamp in the header is the acquisition time of 
                             # the first ray in the scan.
                             #
                             # in frame frame_id, angles are measured around 
                             # the positive Z axis (counterclockwise, if Z is up)
                             # with zero angle being forward along the x axis
                             
    float32 angle_min        # start angle of the scan [rad]
    float32 angle_max        # end angle of the scan [rad]
    float32 angle_increment  # angular distance between measurements [rad]
    
    float32 time_increment   # time between measurements [seconds] - if your scanner
                             # is moving, this will be used in interpolating position
                             # of 3d points
    float32 scan_time        # time between scans [seconds]
    
    float32 range_min        # minimum range value [m]
    float32 range_max        # maximum range value [m]
    
    float32[] ranges         # range data [m] (Note: values < range_min or > range_max should be discarded)
    float32[] intensities    # intensity data [device-specific units].  If your
                             # device does not provide intensities, please leave
                             # the array empty.
    
    ================================================================================
    MSG: std_msgs/Header
    # Standard metadata for higher-level stamped data types.
    # This is generally used to communicate timestamped data 
    # in a particular coordinate frame.
    # 
    # sequence ID: consecutively increasing ID 
    uint32 seq
    #Two-integer timestamp that is expressed as:
    # * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
    # * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
    # time-handling sugar is provided by the client library
    time stamp
    #Frame this data is associated with
    string frame_id
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ClearLaserRequest(null);
    if (msg.weight !== undefined) {
      resolved.weight = msg.weight;
    }
    else {
      resolved.weight = 0
    }

    if (msg.laser !== undefined) {
      resolved.laser = sensor_msgs.msg.LaserScan.Resolve(msg.laser)
    }
    else {
      resolved.laser = new sensor_msgs.msg.LaserScan()
    }

    return resolved;
    }
};

class ClearLaserResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.b = null;
    }
    else {
      if (initObj.hasOwnProperty('b')) {
        this.b = initObj.b
      }
      else {
        this.b = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ClearLaserResponse
    // Serialize message field [b]
    bufferOffset = _serializer.float32(obj.b, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ClearLaserResponse
    let len;
    let data = new ClearLaserResponse(null);
    // Deserialize message field [b]
    data.b = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'custom_msgs/ClearLaserResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'b2fa35766f3759aa2164b5f04811b518';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32 b
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ClearLaserResponse(null);
    if (msg.b !== undefined) {
      resolved.b = msg.b;
    }
    else {
      resolved.b = 0.0
    }

    return resolved;
    }
};

module.exports = {
  Request: ClearLaserRequest,
  Response: ClearLaserResponse,
  md5sum() { return 'f427fbbf8b1198bb81f85f08fbb4a779'; },
  datatype() { return 'custom_msgs/ClearLaser'; }
};
