"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import logging
import tempfile
import unittest
from datetime import datetime, timezone
from packaging.version import parse as parse_version

from picongpu.piccom import Communicator
from picongpu.pypicongpu.simulation_logger.simulation_logger import (
    SimulationLogger,
    logged_operation,
    make_serialisable,
)


class Payload:
    def __init__(self, payload="arbitrary content"):
        self.payload = payload

    def make_serialisable(self):
        return {"payload": make_serialisable(self.payload)}


class DispatchedSerialisable:
    pass


@make_serialisable.register
def make_serialisable_dispatch(obj: DispatchedSerialisable):
    return {"this was": "dispatched!"}


class TestMakeSerialisable(unittest.TestCase):
    def test_payload_serialises_to_dict_with_correct_content(self):
        for p in [1, "asdf", {1: -1, 2: -2, 3: -3}]:
            result = make_serialisable(Payload(p))
            self.assertSequenceEqual({"payload": p}.items(), result.items())

    def test_serialises_default_serialisable(self):
        for p in [1, "asdf", {1: -1, 2: -2, 3: -3}]:
            result = make_serialisable(p)
            self.assertEqual(p, result)

    def test_raises_for_unknown_object(self):
        class NotSerialisable:
            pass

        with self.assertRaisesRegex(ValueError, "I don't know how to make .* of type .* serialisable."):
            make_serialisable(NotSerialisable())

    def test_can_use_dispatch(self):
        instance = DispatchedSerialisable()
        self.assertEqual(make_serialisable(instance), make_serialisable_dispatch(instance))


class TestSimulationLogger(unittest.TestCase):
    def setUp(self) -> None:
        try:
            # On POSIX-compliant linux systems, this would reside in memory.
            # Good for testing because we don't have to go to the physical disc.
            self.storage = tempfile.TemporaryDirectory(dir="/dev/shm")
        except Exception:
            # Not sure what to catch here... sorry!
            # We'll be a bit slower with this:
            logging.warning("Using in-memory directory failed.")
            self.storage = tempfile.TemporaryDirectory()

        self.database = Communicator("arbitrary_author", directory=self.storage.name)
        self.logger = SimulationLogger(self.database)
        self.arbitrary_payload = Payload()

    def tearDown(self) -> None:
        self.storage.cleanup()
        return super().tearDown()


class TestSimulationLoggerLog(TestSimulationLogger):
    def test_logger_can_retrieve_full_content(self):
        identifier = self.logger.log(self.arbitrary_payload)
        self.assertEqual(
            self.logger.get_full_content(identifier), self.database.get_content(self.logger.get_object_id(identifier))
        )

    def test_logger_can_retrieve_full_content_from_object_id(self):
        action_id = self.logger.log(self.arbitrary_payload)
        object_id = self.logger.get_object_id(action_id)
        self.assertEqual(self.logger.get_full_content(object_id), self.logger.get_full_content(action_id))

    def test_logger_full_content_has_metadata_version_information(self):
        identifier = self.logger.log(self.arbitrary_payload)
        # This check is mostly about it being a valid version identifier at all.
        # The parsing would raise an expection if not.
        self.assertGreaterEqual(
            parse_version(self.logger.get_full_content(identifier)["metadata_format_version"]), parse_version("0.1.0")
        )

    def test_logger_can_retrieve_log(self):
        identifier = self.logger.log(self.arbitrary_payload)
        self.assertEqual(self.logger.get_log(identifier), self.logger.get_full_content(identifier)["log"])

    def test_logger_can_retrieve_log_from_object_id(self):
        object_id = self.logger.get_object_id(self.logger.log(self.arbitrary_payload))
        self.assertEqual(self.logger.get_log(object_id), self.logger.get_full_content(object_id)["log"])

    def test_logger_can_retrieve_action(self):
        identifier = self.logger.log(self.arbitrary_payload)
        self.assertEqual(
            self.logger.get_action(identifier),
            next(iter(self.database.get_content(self.logger.get_object_id(identifier))["log"].values())),
        )

    def test_stores_content_in_database(self):
        identifier = self.logger.log(self.arbitrary_payload)
        content = self.logger.get_action(identifier)["content"]
        self.assertEqual(make_serialisable(self.arbitrary_payload), content)

    def test_adds_a_timestamp_of_the_composition_time(self):
        start = datetime.now(timezone.utc)
        identifier = self.logger.log(self.arbitrary_payload)
        end = datetime.now(timezone.utc)
        timestamp = datetime.fromisoformat(self.logger.get_action(identifier)["timestamp"])

        # happened between start and end
        self.assertLessEqual(start, timestamp)
        self.assertLessEqual(timestamp, end)

    def test_adds_an_action(self):
        action = "some action"
        identifier = self.logger.log(self.arbitrary_payload, action=action)
        self.assertEqual(self.logger.get_action(identifier)["action_name"], action)

    def test_adds_an_unspecified_action(self):
        identifier = self.logger.log(self.arbitrary_payload)
        self.assertEqual(self.logger.get_action(identifier)["action_name"], "unspecified")

    def test_freshly_logged_actions_are_no_updates(self):
        identifier = self.logger.log(self.arbitrary_payload)
        self.assertIsNone(self.logger.get_action(identifier)["update_of"])


class TestSimulationLoggerUpdate(TestSimulationLogger):
    def setUp(self):
        super().setUp()
        self.identifier = self.logger.log(self.arbitrary_payload)
        self.update_payload = Payload("update")

    def test_update_has_new_identifier(self):
        new_identifier = self.logger.update(action_id=self.identifier)
        self.assertNotEqual(new_identifier, self.identifier)

    def test_adds_an_action(self):
        self.assertEqual(len(self.logger.get_log(self.identifier)), 1)
        self.logger.update(action_id=self.identifier)
        self.assertEqual(len(self.logger.get_full_content(self.identifier)["log"]), 2)

    def test_added_action_has_correct_name(self):
        action = "A very cool update"
        identifier = self.logger.update(action_id=self.identifier, action=action)
        self.assertEqual(self.logger.get_action(identifier)["action_name"], action)

    def test_added_action_knows_what_it_updated(self):
        identifier = self.logger.update(action_id=self.identifier)
        self.assertEqual(self.logger.get_action(identifier)["update_of"], self.identifier)

    def test_success(self):
        result = self.logger.get_action(self.logger.success(self.identifier))
        self.assertEqual(result["content"]["status"], "success")
        self.assertEqual(result["content"]["info"], None)
        self.assertEqual(result["update_of"], self.identifier)
        self.assertEqual(result["action_name"], "success")

    def test_failure(self):
        exception = ValueError("I messed it up.")
        result = self.logger.get_action(self.logger.failure(self.identifier, exception=exception))
        self.assertEqual(result["content"]["status"], "exception")
        self.assertEqual(result["content"]["exception"], str(exception))
        self.assertEqual(result["content"]["info"], None)
        self.assertEqual(result["update_of"], self.identifier)
        self.assertEqual(result["action_name"], "failure")


class ClassWihLoggedMethods:
    def __init__(self, logger):
        self.logger = logger

    @logged_operation
    def simple_logged_operation(self, x):
        return x + 1

    @logged_operation
    def failing_operation(self):
        raise Exception("Something went wrong.")

    def make_serialisable(self):
        return {"some value": 17}


class TestLoggedOperation(TestSimulationLogger):
    def setUp(self):
        super().setUp()
        self.instance = ClassWihLoggedMethods(self.logger)

    def test_simple_logged_operation(self):
        result = self.instance.simple_logged_operation(42)
        self.assertEqual(result, 43)
        log = self.logger.get_log(self.logger.get_all_objects()[0])
        # Just to be explicit about what's in there:
        self.assertEqual(len(log), len(["action", "success"]))
        self.assertTrue("success" in (val["action_name"] for val in log.values()))

    def test_failing_operation(self):
        try:
            self.instance.failing_operation()
        except Exception:
            # We want this to have failed.
            pass
        log = self.logger.get_log(self.logger.get_all_objects()[0])
        # Just to be explicit about what's in there:
        self.assertEqual(len(log), len(["action", "success"]))
        self.assertTrue("failure" in (val["action_name"] for val in log.values()))

    def test_dependent_operations(self):
        result1, action_id = self.instance.simple_logged_operation(32, return_action_id=True)
        result2 = self.instance.simple_logged_operation(17, update_of=action_id)
        self.assertEqual(result1, 33)
        self.assertEqual(result2, 18)
        log = self.logger.get_log(self.logger.get_all_objects()[0])
        self.assertEqual(len(log), 4)


if __name__ == "__main__":
    unittest.main()
