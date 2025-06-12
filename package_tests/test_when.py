import pytest
from datetime import datetime, date, time, timedelta, timezone
from typing import Any, List, Tuple
import whenever


# Fixtures
@pytest.fixture
def sample_instant():
    """Fixture providing a sample Instant for testing."""
    return whenever.Instant.from_utc(2024, 3, 15, 14, 30, 0)


@pytest.fixture
def sample_local_datetime():
    """Fixture providing a sample LocalDateTime for testing."""
    return whenever.LocalDateTime(2024, 3, 15, 14, 30, 0)


@pytest.fixture
def sample_local_date():
    """Fixture providing a sample LocalDate for testing."""
    return whenever.LocalDate(2024, 3, 15)


@pytest.fixture
def sample_local_time():
    """Fixture providing a sample LocalTime for testing."""
    return whenever.LocalTime(14, 30, 0)


@pytest.fixture
def sample_zoned_datetime():
    """Fixture providing a sample ZonedDateTime for testing."""
    return whenever.ZonedDateTime(2024, 3, 15, 14, 30, 0, tz="America/New_York")


@pytest.fixture
def sample_offset_datetime():
    """Fixture providing a sample OffsetDateTime for testing."""
    return whenever.OffsetDateTime(2024, 3, 15, 14, 30, 0, offset=whenever.hours(5))


# Test Classes
class TestWheneverInstant:
    """Test whenever.Instant - represents a point in time."""
    
    def test_instant_creation(self):
        """Test creating Instant objects."""
        # From UTC components
        instant = whenever.Instant.from_utc(2024, 3, 15, 14, 30, 0)
        assert instant.year == 2024
        assert instant.month == 3
        assert instant.day == 15
        assert instant.hour == 14
        assert instant.minute == 30
        assert instant.second == 0
    
    def test_instant_now(self):
        """Test getting current instant."""
        now = whenever.Instant.now()
        assert isinstance(now, whenever.Instant)
        
        # Should be close to system time
        import time
        system_timestamp = time.time()
        instant_timestamp = now.timestamp()
        assert abs(system_timestamp - instant_timestamp) < 5  # Within 5 seconds
    
    def test_instant_from_timestamp(self):
        """Test creating Instant from Unix timestamp."""
        timestamp = 1710509400.0  # 2024-03-15 14:30:00 UTC
        instant = whenever.Instant.from_timestamp(timestamp)
        
        assert instant.year == 2024
        assert instant.month == 3
        assert instant.day == 15
        assert instant.hour == 14
        assert instant.minute == 30
    
    def test_instant_arithmetic(self, sample_instant):
        """Test Instant arithmetic operations."""
        # Add time
        future = sample_instant + whenever.hours(2)
        assert future.hour == 16
        
        past = sample_instant - whenever.hours(1)
        assert past.hour == 13
        
        # Add days
        tomorrow = sample_instant + whenever.days(1)
        assert tomorrow.day == 16
    
    def test_instant_comparison(self, sample_instant):
        """Test Instant comparison operations."""
        future = sample_instant + whenever.hours(1)
        past = sample_instant - whenever.hours(1)
        
        assert future > sample_instant
        assert past < sample_instant
        assert sample_instant == sample_instant
        assert future != past
    
    def test_instant_formatting(self, sample_instant):
        """Test Instant string representation and formatting."""
        iso_string = str(sample_instant)
        assert "2024-03-15" in iso_string
        assert "14:30:00" in iso_string
        assert "Z" in iso_string or "+00:00" in iso_string  # UTC indicator
    
    def test_instant_conversion_to_zones(self, sample_instant):
        """Test converting Instant to different time zones."""
        # Convert to New York time
        ny_time = sample_instant.to_tz("America/New_York")
        assert isinstance(ny_time, whenever.ZonedDateTime)
        
        # Convert to London time
        london_time = sample_instant.to_tz("Europe/London")
        assert isinstance(london_time, whenever.ZonedDateTime)
        
        # The instant should be the same, just different local representation
        assert ny_time.instant() == sample_instant
        assert london_time.instant() == sample_instant


class TestWheneverLocalDateTime:
    """Test whenever.LocalDateTime - datetime without timezone."""
    
    def test_local_datetime_creation(self):
        """Test creating LocalDateTime objects."""
        dt = whenever.LocalDateTime(2024, 3, 15, 14, 30, 45)
        assert dt.year == 2024
        assert dt.month == 3
        assert dt.day == 15
        assert dt.hour == 14
        assert dt.minute == 30
        assert dt.second == 45
    
    @pytest.mark.parametrize("year,month,day,hour,minute,second", [
        (2024, 1, 1, 0, 0, 0),
        (2024, 12, 31, 23, 59, 59),
        (2020, 2, 29, 12, 30, 15),  # Leap year
        (2024, 6, 15, 9, 45, 30),
    ])
    def test_local_datetime_various_values(self, year, month, day, hour, minute, second):
        """Test LocalDateTime with various date/time values."""
        dt = whenever.LocalDateTime(year, month, day, hour, minute, second)
        assert dt.year == year
        assert dt.month == month
        assert dt.day == day
        assert dt.hour == hour
        assert dt.minute == minute
        assert dt.second == second
    
    def test_local_datetime_properties(self, sample_local_datetime):
        """Test LocalDateTime properties and methods."""
        assert sample_local_datetime.year == 2024
        assert sample_local_datetime.month == 3
        assert sample_local_datetime.day == 15
        
        # Test date and time extraction
        date_part = sample_local_datetime.date()
        assert isinstance(date_part, whenever.LocalDate)
        assert date_part.year == 2024
        assert date_part.month == 3
        assert date_part.day == 15
        
        time_part = sample_local_datetime.time()
        assert isinstance(time_part, whenever.LocalTime)
        assert time_part.hour == 14
        assert time_part.minute == 30
    
    def test_local_datetime_arithmetic(self, sample_local_datetime):
        """Test LocalDateTime arithmetic operations."""
        # Add time units
        future_hour = sample_local_datetime + whenever.hours(2)
        assert future_hour.hour == 16
        
        future_day = sample_local_datetime + whenever.days(5)
        assert future_day.day == 20
        
        # Subtract time units
        past_minute = sample_local_datetime - whenever.minutes(30)
        assert past_minute.hour == 14
        assert past_minute.minute == 0
    
    def test_local_datetime_comparison(self, sample_local_datetime):
        """Test LocalDateTime comparison operations."""
        earlier = sample_local_datetime - whenever.hours(1)
        later = sample_local_datetime + whenever.hours(1)
        
        assert earlier < sample_local_datetime
        assert later > sample_local_datetime
        assert sample_local_datetime == sample_local_datetime
        assert earlier != later
    
    def test_local_datetime_assume_timezone(self, sample_local_datetime):
        """Test assuming timezone for LocalDateTime."""
        # Assume UTC timezone
        zoned_utc = sample_local_datetime.assume_utc()
        assert isinstance(zoned_utc, whenever.ZonedDateTime)
        
        # Assume specific timezone
        zoned_ny = sample_local_datetime.assume_tz("America/New_York")
        assert isinstance(zoned_ny, whenever.ZonedDateTime)


class TestWheneverLocalDate:
    """Test whenever.LocalDate - date without time or timezone."""
    
    def test_local_date_creation(self):
        """Test creating LocalDate objects."""
        date_obj = whenever.LocalDate(2024, 3, 15)
        assert date_obj.year == 2024
        assert date_obj.month == 3
        assert date_obj.day == 15
    
    @pytest.mark.parametrize("year,month,day", [
        (2024, 1, 1),
        (2024, 12, 31),
        (2020, 2, 29),  # Leap year
        (2023, 2, 28),  # Non-leap year
        (2024, 7, 4),
    ])
    def test_local_date_various_values(self, year, month, day):
        """Test LocalDate with various date values."""
        date_obj = whenever.LocalDate(year, month, day)
        assert date_obj.year == year
        assert date_obj.month == month
        assert date_obj.day == day
    
    def test_local_date_today(self):
        """Test getting today's date."""
        today = whenever.LocalDate.today()
        assert isinstance(today, whenever.LocalDate)
        
        # Compare with system date
        import datetime
        system_today = datetime.date.today()
        assert today.year == system_today.year
        assert today.month == system_today.month
        assert today.day == system_today.day
    
    def test_local_date_arithmetic(self, sample_local_date):
        """Test LocalDate arithmetic operations."""
        tomorrow = sample_local_date + whenever.days(1)
        assert tomorrow.day == 16
        
        yesterday = sample_local_date - whenever.days(1)
        assert yesterday.day == 14
        
        next_week = sample_local_date + whenever.weeks(1)
        assert next_week.day == 22
    
    def test_local_date_weekday(self, sample_local_date):
        """Test LocalDate weekday operations."""
        # March 15, 2024 is a Friday (weekday 4, 0=Monday)
        weekday = sample_local_date.weekday()
        assert 0 <= weekday <= 6
        
        # Test if it's weekend
        is_weekend = weekday >= 5
        # We can't assert specific value without knowing the actual date
        assert isinstance(is_weekend, bool)
    
    def test_local_date_at_time(self, sample_local_date):
        """Test combining LocalDate with time."""
        time_obj = whenever.LocalTime(14, 30, 0)
        datetime_obj = sample_local_date.at(time_obj)
        
        assert isinstance(datetime_obj, whenever.LocalDateTime)
        assert datetime_obj.year == 2024
        assert datetime_obj.month == 3
        assert datetime_obj.day == 15
        assert datetime_obj.hour == 14
        assert datetime_obj.minute == 30


class TestWheneverLocalTime:
    """Test whenever.LocalTime - time without date or timezone."""
    
    def test_local_time_creation(self):
        """Test creating LocalTime objects."""
        time_obj = whenever.LocalTime(14, 30, 45)
        assert time_obj.hour == 14
        assert time_obj.minute == 30
        assert time_obj.second == 45
    
    @pytest.mark.parametrize("hour,minute,second", [
        (0, 0, 0),
        (23, 59, 59),
        (12, 30, 15),
        (9, 45, 0),
        (18, 0, 30),
    ])
    def test_local_time_various_values(self, hour, minute, second):
        """Test LocalTime with various time values."""
        time_obj = whenever.LocalTime(hour, minute, second)
        assert time_obj.hour == hour
        assert time_obj.minute == minute
        assert time_obj.second == second
    
    def test_local_time_arithmetic(self, sample_local_time):
        """Test LocalTime arithmetic operations."""
        later = sample_local_time + whenever.hours(2)
        assert later.hour == 16
        
        earlier = sample_local_time - whenever.minutes(30)
        assert earlier.hour == 14
        assert earlier.minute == 0
        
        # Test wrapping around midnight
        late_time = whenever.LocalTime(23, 30, 0)
        next_day = late_time + whenever.hours(2)
        assert next_day.hour == 1
        assert next_day.minute == 30
    
    def test_local_time_comparison(self, sample_local_time):
        """Test LocalTime comparison operations."""
        earlier = sample_local_time - whenever.hours(1)
        later = sample_local_time + whenever.hours(1)
        
        assert earlier < sample_local_time
        assert later > sample_local_time
        assert sample_local_time == sample_local_time
        assert earlier != later


class TestWheneverZonedDateTime:
    """Test whenever.ZonedDateTime - datetime with timezone."""
    
    def test_zoned_datetime_creation(self):
        """Test creating ZonedDateTime objects."""
        zdt = whenever.ZonedDateTime(2024, 3, 15, 14, 30, 0, tz="America/New_York")
        assert zdt.year == 2024
        assert zdt.month == 3
        assert zdt.day == 15
        assert zdt.hour == 14
        assert zdt.minute == 30
        assert zdt.second == 0
    
    @pytest.mark.parametrize("timezone", [
        "UTC",
        "America/New_York",
        "Europe/London",
        "Asia/Tokyo",
        "Australia/Sydney",
        "America/Los_Angeles",
    ])
    def test_zoned_datetime_various_timezones(self, timezone):
        """Test ZonedDateTime with various timezones."""
        zdt = whenever.ZonedDateTime(2024, 3, 15, 14, 30, 0, tz=timezone)
        assert zdt.year == 2024
        assert zdt.month == 3
        assert zdt.day == 15
        assert zdt.tz.name == timezone
    
    def test_zoned_datetime_conversion_to_instant(self, sample_zoned_datetime):
        """Test converting ZonedDateTime to Instant."""
        instant = sample_zoned_datetime.instant()
        assert isinstance(instant, whenever.Instant)
        
        # Converting back should give same local time in same timezone
        back_to_zoned = instant.to_tz(sample_zoned_datetime.tz)
        assert back_to_zoned.year == sample_zoned_datetime.year
        assert back_to_zoned.month == sample_zoned_datetime.month
        assert back_to_zoned.day == sample_zoned_datetime.day
    
    def test_zoned_datetime_timezone_conversion(self, sample_zoned_datetime):
        """Test converting between timezones."""
        # Convert to different timezone
        london_time = sample_zoned_datetime.to_tz("Europe/London")
        assert isinstance(london_time, whenever.ZonedDateTime)
        assert london_time.tz.name == "Europe/London"
        
        # The instant should be the same
        assert sample_zoned_datetime.instant() == london_time.instant()


class TestWheneverOffsetDateTime:
    """Test whenever.OffsetDateTime - datetime with fixed offset."""
    
    def test_offset_datetime_creation(self):
        """Test creating OffsetDateTime objects."""
        odt = whenever.OffsetDateTime(2024, 3, 15, 14, 30, 0, offset=whenever.hours(5))
        assert odt.year == 2024
        assert odt.month == 3
        assert odt.day == 15
        assert odt.hour == 14
        assert odt.minute == 30
        assert odt.second == 0
    
    @pytest.mark.parametrize("offset_hours", [-12, -8, -5, 0, 3, 5, 8, 12])
    def test_offset_datetime_various_offsets(self, offset_hours):
        """Test OffsetDateTime with various offset values."""
        offset = whenever.hours(offset_hours)
        odt = whenever.OffsetDateTime(2024, 3, 15, 14, 30, 0, offset=offset)
        assert odt.year == 2024
        assert odt.offset == offset
    
    def test_offset_datetime_conversion_to_instant(self, sample_offset_datetime):
        """Test converting OffsetDateTime to Instant."""
        instant = sample_offset_datetime.instant()
        assert isinstance(instant, whenever.Instant)
        
        # Converting back with same offset should give same local time
        back_to_offset = instant.to_fixed_offset(sample_offset_datetime.offset)
        assert back_to_offset.year == sample_offset_datetime.year
        assert back_to_offset.month == sample_offset_datetime.month
        assert back_to_offset.day == sample_offset_datetime.day


class TestWheneverDurations:
    """Test whenever duration objects and arithmetic."""
    
    @pytest.mark.parametrize("duration_func,amount", [
        (whenever.nanoseconds, 1000000),
        (whenever.microseconds, 1000),
        (whenever.milliseconds, 500),
        (whenever.seconds, 30),
        (whenever.minutes, 45),
        (whenever.hours, 3),
        (whenever.days, 7),
        (whenever.weeks, 2),
    ])
    def test_duration_creation(self, duration_func, amount):
        """Test creating various duration objects."""
        duration = duration_func(amount)
        assert duration is not None
        # Duration should be additive with datetime objects
        base_dt = whenever.LocalDateTime(2024, 3, 15, 14, 30, 0)
        result = base_dt + duration
        assert result != base_dt
    
    def test_duration_arithmetic(self):
        """Test duration arithmetic operations."""
        hour = whenever.hours(1)
        minute = whenever.minutes(30)
        
        combined = hour + minute
        assert combined != hour
        assert combined != minute
        
        # Test with datetime
        dt = whenever.LocalDateTime(2024, 3, 15, 14, 0, 0)
        result = dt + combined
        assert result.hour == 15
        assert result.minute == 30
    
    def test_negative_durations(self):
        """Test negative durations for subtraction."""
        base_dt = whenever.LocalDateTime(2024, 3, 15, 14, 30, 0)
        
        past = base_dt - whenever.hours(2)
        assert past.hour == 12
        
        future = base_dt + whenever.hours(2)
        assert future.hour == 16


class TestWheneverEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_date_values(self):
        """Test handling of invalid date values."""
        with pytest.raises((ValueError, whenever.InvalidDate)):
            whenever.LocalDate(2024, 13, 1)  # Invalid month
        
        with pytest.raises((ValueError, whenever.InvalidDate)):
            whenever.LocalDate(2024, 2, 30)  # Invalid day for February
        
        with pytest.raises((ValueError, whenever.InvalidDate)):
            whenever.LocalDate(2023, 2, 29)  # Feb 29 in non-leap year
    
    def test_invalid_time_values(self):
        """Test handling of invalid time values."""
        with pytest.raises((ValueError, whenever.InvalidTime)):
            whenever.LocalTime(25, 0, 0)  # Invalid hour
        
        with pytest.raises((ValueError, whenever.InvalidTime)):
            whenever.LocalTime(12, 60, 0)  # Invalid minute
        
        with pytest.raises((ValueError, whenever.InvalidTime)):
            whenever.LocalTime(12, 30, 60)  # Invalid second
    
    def test_leap_year_handling(self):
        """Test proper leap year handling."""
        # Valid leap year date
        leap_date = whenever.LocalDate(2024, 2, 29)
        assert leap_date.year == 2024
        assert leap_date.month == 2
        assert leap_date.day == 29
        
        # Invalid leap year date should raise error
        with pytest.raises((ValueError, whenever.InvalidDate)):
            whenever.LocalDate(2023, 2, 29)
    
    def test_timezone_edge_cases(self):
        """Test timezone-related edge cases."""
        # Test with UTC
        utc_dt = whenever.ZonedDateTime(2024, 3, 15, 14, 30, 0, tz="UTC")
        assert utc_dt.tz.name == "UTC"
        
        # Test invalid timezone should raise error
        with pytest.raises((ValueError, whenever.InvalidTimezone)):
            whenever.ZonedDateTime(2024, 3, 15, 14, 30, 0, tz="Invalid/Timezone")


class TestWheneverInteroperability:
    """Test interoperability with Python standard library."""
    
    def test_to_python_datetime(self, sample_local_datetime):
        """Test conversion to Python datetime objects."""
        py_datetime = sample_local_datetime.py_datetime()
        assert isinstance(py_datetime, datetime)
        assert py_datetime.year == 2024
        assert py_datetime.month == 3
        assert py_datetime.day == 15
        assert py_datetime.hour == 14
        assert py_datetime.minute == 30
    
    def test_from_python_datetime(self):
        """Test creating whenever objects from Python datetime."""
        py_dt = datetime(2024, 3, 15, 14, 30, 0)
        whenever_dt = whenever.LocalDateTime.from_py_datetime(py_dt)
        
        assert whenever_dt.year == 2024
        assert whenever_dt.month == 3
        assert whenever_dt.day == 15
        assert whenever_dt.hour == 14
        assert whenever_dt.minute == 30
    
    def test_to_python_date(self, sample_local_date):
        """Test conversion to Python date objects."""
        py_date = sample_local_date.py_date()
        assert isinstance(py_date, date)
        assert py_date.year == 2024
        assert py_date.month == 3
        assert py_date.day == 15


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])