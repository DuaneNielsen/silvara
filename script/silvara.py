#!/usr/bin/env python3
"""
 * File: offb_node.py
 * Stack and tested in Gazebo 9 SITL
"""

import math
import roslib

roslib.load_manifest('zed_interfaces')
from zed_interfaces.msg import ObjectsStamped

from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest

import argparse
from pathlib import Path
import threading
import time
import sys
import traceback
from collections import deque
import numpy as np
import cv2
import torch
import torch.nn as nn

# ROS
import rospy

from std_msgs.msg import Int16
# MAVROS
from mavros_msgs.msg import OverrideRCIn, RCIn, RCOut
# from mavros_msgs.srv import CommandBool
from mavros_msgs.msg import WaypointList, WaypointReached

from mavros_msgs.srv import WaypointPush, WaypointPull, WaypointClear, WaypointSetCurrent
# from mavros_msgs.srv import *
from mavros_msgs.srv import ParamGet

from mavros_msgs.srv import CommandLong

# from mavros_msgs.srv import ParamSet, SetMode
# TODO Missing import from mavros_msgs.srv import WaypointGOTO
# from mavros.mission import *
current_state = State()


def state_cb(msg):
    global current_state
    current_state = msg


# Globals
THROTTLE_CHANNEL = 2
STEER_CHANNEL = 0

EXEC_TIME = 1  # exc time in secs


def get_servo_pwm(pos):
    """ takes a number in range -1.0 .. 1.0 and converts to the servo pulse width"""
    pwm = int(rospy.get_param('/SERVO_RANGE') * pos + rospy.get_param('/SERVO_CENTER') + rospy.get_param('/SERVO_TRIM'))
    pwm = min(rospy.get_param('/SERVO_MAX'), pwm)
    pwm = max(rospy.get_param('/SERVO_MIN'), pwm)
    return pwm


def get_throttle_pwm(pos):
    """ takes a number in range -1.0 .. 1.0 and converts to the servo pulse width"""
    pwm = int(rospy.get_param('/THROTTLE_RANGE') * pos + rospy.get_param('/THROTTLE_CENTER') + rospy.get_param('/THROTTLE_TRIM'))
    pwm = min(rospy.get_param('/THROTTLE_MAX'), pwm)
    pwm = max(rospy.get_param('/THROTTLE_MIN'), pwm)
    return pwm


class UAV_Control:
    """UAV WP and Manual Control"""

    def __init__(self):
        self.lock = threading.Lock()
        # mavros.set_namespace("/mavros")
        self.waypoint_list = None
        self.current_waypoint = None

        # Proxies
        rospy.wait_for_service('/mavros/param/get')
        self.svc_get_param = rospy.ServiceProxy('/mavros/param/get', ParamGet)

        rospy.wait_for_service('/mavros/mission/push')
        self.svc_push_waypoints = rospy.ServiceProxy('/mavros/mission/push', WaypointPush)

        rospy.wait_for_service('/mavros/mission/pull')
        self.svc_pull_waypoints = rospy.ServiceProxy('/mavros/mission/pull', WaypointPull)

        rospy.wait_for_service('/mavros/mission/clear')
        self.svc_clear_waypoints = rospy.ServiceProxy('mavros/mission/clear', WaypointClear)

        rospy.wait_for_service('/mavros/mission/set_current')
        self.svc_set_current_waypoint = rospy.ServiceProxy(
            'mavros/mission/set_current',
            WaypointSetCurrent)

        rospy.wait_for_service('/mavros/cmd/command')
        self._srv_cmd_long = rospy.ServiceProxy(
            '/mavros/cmd/command', CommandLong, persistent=True)

        # Publishers
        self.pub_rc_override = rospy.Publisher('mavros/rc/override', OverrideRCIn, queue_size=10)

        # Subscribers
        self.sub_waypoints = rospy.Subscriber("/mavros/mission/waypoints", WaypointList, self.__waypoints_cb)
        self.sub_current = rospy.Subscriber("/mavros/mission/reached", WaypointReached, self.__current_cb)
        self.sub_rc_in = rospy.Subscriber("/mavros/rc/in", RCIn, self.__rcin_cb)
        self.sub_rc_out = rospy.Subscriber("/mavros/rc/out", RCOut, self.__rcout_cb)
        self.servo = 1500
        self.throttle= 1000

    def __waypoints_cb(self, topic):
        self.lock.acquire()
        try:
            self.waypoint_list = topic.waypoints
        finally:
            self.lock.release()

    def __current_cb(self, waypoint_reached):
        rospy.loginfo('__current_cb: ')
        rospy.loginfo('__current_cb: %d', waypoint_reached.wp_seq)
        self.lock.acquire()
        try:
            self.current_waypoint = waypoint_reached.wp_seq
            wp = self.waypoint_list[self.current_waypoint]
            cone_alt = wp.z_alt
            (q, r) = divmod(cone_alt, 2)
            if r == 1:
                rospy.set_param("/CONE_ON_GRASS", True)
                rospy.loginfo('Cone is on grass')
            else:
                rospy.set_param("/CONE_ON_GRASS", False)
                rospy.loginfo('Cone is not on grass')
        except:
            rospy.loginfo("Failed to get current waypoint details")
            # Make a safe bet
            rospy.set_param("/CONE_ON_GRASS", True)
        finally:
            self.lock.release()

    def __rcin_cb(self, rc_in):
        throttle = rc_in.channels[THROTTLE_CHANNEL]
        self.set_throttle_servo(self.throttle, self.servo)
        # rospy.loginfo(rc_in)

    def __rcout_cb(self, rc_out):
        pass
        #rospy.loginfo(rc_out)

    def print_waypoints(self):
        """Prints Pixhawk waypoints to stdout"""
        for seq, waypoint in enumerate(self.waypoint_list):
            print(' seq: ' + str(seq) +
                  ' waypoint.is_current: ' + str(waypoint.is_current) +
                  ' waypoint.autocontinue: ' + str(waypoint.autocontinue) +
                  ' waypoint.frame: ' + str(waypoint.frame) +
                  ' waypoint.command: ' + str(waypoint.command) +
                  ' waypoint.param1: ' + str(waypoint.param1) +
                  ' waypoint.param2: ' + str(waypoint.param2) +
                  ' waypoint.param3: ' + str(waypoint.param3) +
                  ' waypoint.param4: ' + str(waypoint.param4) +
                  ' waypoint.x_lat: ' + str(waypoint.x_lat) +
                  ' waypoint.y_long: ' + str(waypoint.y_long) +
                  ' waypoint.z_alt: ' + str(waypoint.z_alt) +
                  '')

    #
    # throttle: Desired PWM value
    #
    def set_throttle(self, throttle):
        """Set throttle"""
        # rospy.loginfo('mavros/rc/override, throttle')
        msg = OverrideRCIn()
        msg.channels[THROTTLE_CHANNEL] = throttle  # Desired PWM value
        # rospy.loginfo(msg)
        self.pub_rc_override.publish(msg)

    #
    # servo: Desired PWM value
    #
    def set_servo(self, servo):
        """Set servo"""
        # rospy.loginfo('mavros/rc/override, servo')
        msg = OverrideRCIn()
        msg.channels[STEER_CHANNEL] = servo  # Desired PWM value
        # rospy.loginfo(msg)
        self.pub_rc_override.publish(msg)

    #
    # throttle: Desired PWM value
    # servo: Desired PWM value
    #
    def set_throttle_servo(self, throttle, servo):
        """Set throttle AND servo"""
        # rospy.loginfo('mavros/rc/override, throttle and servo')
        msg = OverrideRCIn()
        if current_state.mode == "MANUAL":
            msg.channels[THROTTLE_CHANNEL] = throttle  # Desired PWM value
            msg.channels[STEER_CHANNEL] = servo  # Desired PWM value
        rospy.loginfo(msg)
        self.pub_rc_override.publish(msg)

    #
    # Push waypoints
    #
    def push_waypoints(self, waypoints):
        """Push waypoints to Pixhawk"""
        rospy.loginfo('/mavros/mission/push')
        try:
            resp = self.svc_push_waypoints(waypoints)
            rospy.loginfo(resp)
            return resp
        except rospy.ServiceException as err:
            rospy.loginfo(
                "Service push_waypoints call failed: %s.",
                err)
            return None

    #
    # Pull waypoints
    # Request update waypoint list.
    #
    def pull_waypoints(self):
        """Request update waypoint list"""
        rospy.loginfo('/mavros/mission/pull')
        try:
            resp = self.svc_pull_waypoints()
            rospy.loginfo('success: ' + str(resp.success) + ' wp_received: ' + str(resp.wp_received))
            return resp
        except rospy.ServiceException as err:
            rospy.loginfo(
                "Service pull_waypoints call failed: %s.",
                err)
            return None

    #
    # Clear waypoints
    #
    def clear_waypoints(self):
        """Clear waypoints"""
        rospy.loginfo('/mavros/mission/clear')
        try:
            resp = self.svc_clear_waypoints()
            rospy.loginfo(resp)
            return resp
        except rospy.ServiceException as err:
            rospy.loginfo(
                "Service clear_waypoints call failed: %s.",
                err)
            return None

    #
    # Set current waypoint
    #
    def set_current_waypoint(self, idx):
        """Set current wp"""
        rospy.loginfo('/mavros/mission/set_current: ' + str(idx))
        try:
            resp = self.svc_set_current_waypoint(idx)
            rospy.loginfo(resp)
            return resp
        except rospy.ServiceException as err:
            rospy.loginfo(
                "Service set_current_waypoint call failed: %s. Index %d could not be set. "
                "Check that GPS is enabled.",
                err, idx)
            return None

    #
    # Goto wp
    #
    #    def goto_waypoint(self, args):
    #        """Go to WP"""
    #        wp = Waypoint(
    #            frame=args.frame,
    #            command=args.command,
    #            param1=args.param1,
    #            param2=args.param2,
    #            param3=args.param3,
    #            param4=args.param4,
    #            x_lat=args.x_lat,
    #            y_long=args.y_long,
    #            z_alt=args.z_alt
    #        )
    #        try:
    #            service = rospy.ServiceProxy('mavros/mission/goto', WaypointGOTO)
    #            resp = service(waypoint=wp)
    #            rospy.loginfo(resp)
    #            return resp
    #        except rospy.ServiceException, e:
    #            rospy.loginfo('Service call failed: {0}'.format(e))
    #            return None

    def get_param_int(self, name):
        """Get parameter value from UAV"""
        ret = None
        try:
            ret = self.svc_get_param(param_id=name)
            return ret.value.integer
        except rospy.ServiceException as ex:
            rospy.logerr(ex)
            return None

    def send_mavros_cmd(self, bool1, msgid, bool2, p0, p1, p2, p3, p4, p5, p6):
        """Send a mavros command"""
        # rospy.loginfo("/mavros/cmd/command/ %s %s %s %s %s %s %s %s %s %s",
        #              str(bool1), str(msgid),
        #              str(bool2), str(p0),
        #              str(p1), str(p2), str(p3), str(p4), str(p5), str(p6))
        self._srv_cmd_long(bool1, msgid, bool2, p0, p1, p2, p3, p4, p5, p6)


class ROSLogger:
    def write(self, message):
        rospy.logdebug(message)


def redirect_stdout():
    # Save a reference to the original stdout
    original_stdout = sys.stdout

    # Create a new stdout that writes to the ROS logging system
    sys.stdout = ROSLogger()

    return original_stdout


def restore_stdout(original_stdout):
    # Reset stdout to the original value
    sys.stdout = original_stdout

class LogInfoMessage:
    def __init__(self, cooldown=0.0):
        self.cooldown = rospy.Duration(cooldown)
        self.last_logged = rospy.Time.now() - self.cooldown

    def __call__(self, log_mesg):
        if rospy.Time.now() - self.last_logged > self.cooldown:
            rospy.loginfo(log_mesg)
            self.last_logged = rospy.Time.now()


if __name__ == "__main__":

    print(f"{sys.argv}")

    rospy.init_node("offb_node_py")

    state_sub = rospy.Subscriber("/mavros/state", State, callback=state_cb)
    rospy.loginfo('Subscribed /mavros/state')

    local_pos_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)
    rospy.loginfo('Published /mavros/state')

    rospy.loginfo('Waiting /mavros/cmd/arming')
    rospy.wait_for_service("mavros/cmd/arming")
    arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

    rospy.loginfo('Waiting /mavros/set_mode')
    rospy.wait_for_service("mavros/set_mode")
    set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

    original_stdout = redirect_stdout()

    ctrl = UAV_Control()

    # PID queue
    angle_to_object = deque(maxlen=rospy.get_param('SERVO_PID_LEN'))
    dist_to_object = deque(maxlen=rospy.get_param('THROTTLE_PID_LEN'))

    tracking_map = {0: "OFF", 1: "OK", 2: "SEARCHING", 3: "TERMINATE", 4:"LAST"}

    in_features = 64 * 3
    mlp_size = 4

    net = nn.Sequential(
        nn.Linear(in_features, mlp_size),
        nn.BatchNorm1d(mlp_size),
        nn.ReLU(inplace=True),
        nn.Linear(mlp_size, mlp_size),
        nn.BatchNorm1d(mlp_size),
        nn.ReLU(inplace=True),
        nn.Linear(mlp_size, 1, bias=False),
        nn.Sigmoid()
    )

    net.load_state_dict(torch.load('/home/duane/color_mlp_weights.pt'))
    net.eval()

    def format_track(track):
        return f"id: {track.label_id} conf: {track.confidence:.2f} " \
               f"tracking_state: {tracking_map[track.tracking_state]} " \
               f"pos: {track.position[0]:.2f}, {track.position[1]:.2f}, {track.position[2]:.2f}" \
               f" vel:{track.velocity[0]:.2f}, {track.velocity[1]:.2f}, {track.velocity[2]:.2f}"

    log_best_track = LogInfoMessage(cooldown=1.0)
    log_angle_to_cone = LogInfoMessage(cooldown=1.0)

    def get_area(o):
        top_left = o.bounding_box_2d.corners[0]
        bottom_right = o.bounding_box_2d.corners[2]
        area = (bottom_right.kp[0] - top_left.kp[0]) * (bottom_right.kp[1] - top_left.kp[1])
        return area

    def get_color_confidence(o):
        red = torch.tensor(o.hist_red)
        green = torch.tensor(o.hist_green)
        blue = torch.tensor(o.hist_blue)
        red = red / red.sum()
        green = green / green.sum()
        blue = blue / blue.sum()
        histo = torch.cat([red, green, blue]).unsqueeze(0)
        color_confidence = net(histo)
        return color_confidence.item()

    def onObjectDetect(objectsStamped):

        best_track = None
        start_time = time.time()
        largest_area = None
        for o in objectsStamped.objects:
            # rospy.loginfo(format_track(o))
            # filter conditions
            area = get_area(o)
            color_confidence = get_color_confidence(o)
            if o.confidence > 50. and o.tracking_state == 0 \
                and not math.isnan(o.position[0]) \
                and area > rospy.get_param('/MIN_OBJECT_AREA') \
                and color_confidence > rospy.get_param('/COLOR_CONFIDENCE'):

                # take track with highest confidence
                if best_track is None:
                    best_track = o
                    largest_area = area
                elif area > largest_area:
                    best_track = o

            if best_track is not None:
                log_best_track(f'color_confidence: {color_confidence} confidence: {o.confidence} area: {area}')

            

        """
        best_track.position[0] -> distance to object (meters), [1] -> distance left/right (meters), [2] up/down (meters)
        """
        if best_track is not None:

            a_to_obj = math.atan(best_track.position[1] / (best_track.position[0] + 0.1) )
            log_angle_to_cone(f'angle_to_cone {a_to_obj}')
            angle_to_object.appendleft(a_to_obj)

            P = angle_to_object[0]
            I = sum(angle_to_object)
            D = angle_to_object[1] - angle_to_object[0] if len(angle_to_object) > 1 else 0.

            pos = P * rospy.get_param('/SERVO_P') \
                  + max(rospy.get_param('/SERVO_IMAX'), I * rospy.get_param('/SERVO_I')) \
                  + D * rospy.get_param('/SERVO_D')
            #ctrl.set_servo(get_servo_pwm(pos))
            ctrl.servo = get_servo_pwm(pos)

            dist_to_object.appendleft(best_track.position[0] - 0.2)

            P = dist_to_object[0]
            I = sum(dist_to_object)
            D = dist_to_object[1] - dist_to_object[0] if len(dist_to_object) > 1 else 0.

            t_pos = P * rospy.get_param('/THROTTLE_P') \
                  + max(rospy.get_param('/THROTTLE_IMAX'), I * rospy.get_param('/THROTTLE_I')) \
                  + D * rospy.get_param('/THROTTLE_D')

            ctrl.throttle = get_throttle_pwm(t_pos)
            ctrl.set_throttle_servo(get_throttle_pwm(pos), get_servo_pwm(pos))

            rate.sleep()


        best_track = None


    zed_obj_det = rospy.Subscriber('/zed2i/zed_node/obj_det/objects', ObjectsStamped, callback=onObjectDetect, queue_size=1)

    # init zed camera and detector thread
    parser = argparse.ArgumentParser()
    rospy.loginfo(f'offboard arggs: {sys.argv}')
    args = parser.parse_args(sys.argv[4:])

    # Setpoint publishing MUST be faster than 2Hz
    rate = rospy.Rate(200)
    i = 0

    # Wait for Flight Controller connection
    while (not rospy.is_shutdown() and not current_state.connected):
        print(current_state.connected)
        rate.sleep()

    pose = PoseStamped()

    pose.pose.position.x = 0
    pose.pose.position.y = 0
    pose.pose.position.z = 2

    rospy.loginfo("sending dummy setpoints")

    # Send a few setpoints before starting
    for i in range(100):
        if (rospy.is_shutdown()):
            break

        local_pos_pub.publish(pose)
        rate.sleep()

    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'MANUAL'

    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True

    last_req = rospy.Time.now()
    last_log_mesg = rospy.Time.now() - rospy.Duration(30.0)
    log_armed_message = LogInfoMessage(cooldown=30.0)

    while (not rospy.is_shutdown()):

        try:
            if (current_state.mode != "MANUAL" and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
                rospy.loginfo('setting mode')
                rospy.loginfo(f'mode: {current_state.mode}')
                rospy.loginfo(f'armed: {current_state.armed}')
                if (set_mode_client.call(offb_set_mode).mode_sent == True):
                    rospy.loginfo("MANUAL enabled")

                last_req = rospy.Time.now()
            else:

                if (not current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
                    rospy.loginfo('arming')
                    rospy.loginfo(f'mode: {current_state.mode}')
                    rospy.loginfo(f'armed: {current_state.armed}')
                if (arming_client.call(arm_cmd).success == True):
                    log_armed_message("Vehicle armed")

                last_req = rospy.Time.now()
        # print(pose)
        # local_pos_pub.publish(pose)

                i += 1
                rate.sleep()
        except Exception as e:
            rospy.logerr(e)
            rospy.logerr(traceback.format_exc())
