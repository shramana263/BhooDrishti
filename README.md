# BhooDrishti - Satellite Change Detection & Monitoring System

## Project Overview

BhooDrishti is a robust change detection, monitoring, and alert system for user-defined Areas of Interest (AOI) using multi-temporal satellite imagery. The system enables automated monitoring of large geographical areas and alerts users to significant changes based on their specified criteria.

## Problem Statement

Satellite imagery is captured daily through dedicated satellites, covering vast areas with high frequency (every 5-10 days). This presents an opportunity for regular monitoring and automated change detection for:

- **Ecological Monitoring**: Forest deforestation detection
- **Legal Compliance**: Illegal land occupation monitoring
- **Infrastructure Protection**: Building encroachment on protected areas
- **Environmental Conservation**: Water body reclamation monitoring

## System Architecture

### Core Components

1. **User Interface Layer**
   - Web-based platform (no software installation required)
   - Interactive map for AOI definition
   - User account management
   - Alert configuration dashboard

2. **Data Processing Engine**
   - Satellite data acquisition from Bhuvan/Google Earth Engine
   - Multi-temporal image preprocessing
   - Cloud and shadow masking
   - Change detection algorithms

3. **Alert System**
   - Automated change event generation
   - Configurable confidence thresholds
   - Multi-channel notification delivery

4. **Export Module**
   - GIS-compatible format support (GeoJSON, Shapefiles)
   - Report generation and visualization

## Key Features

### User-Defined AOI Management
- Interactive map-based AOI drawing
- Multiple AOI support per user account
- AOI persistence and management
- Customizable monitoring parameters

### Intelligent Change Detection
- Multi-temporal satellite imagery analysis
- Before/after image comparison
- Automated GIS information processing
- False positive reduction through cloud/shadow masking

### Flexible Alert Configuration
- Customizable confidence thresholds (30% - 90%)
- Multiple alert types:
  - Deforestation events
  - Building construction
  - Water body reclamation
  - Land occupation changes

### Data Sources
- **Primary**: Bhuvan satellite data
- **Alternative**: Google Earth Engine APIs
- **Processing**: Automated download and preprocessing

## Technical Implementation

### Data Processing Pipeline

```
Satellite Data → Preprocessing → Change Detection → Alert Generation → User Notification
```

1. **Data Acquisition**
   - Automated satellite data download
   - Historical and current imagery retrieval
   - Multi-spectral band processing

2. **Preprocessing**
   - Cloud cover detection and masking
   - Shadow removal algorithms
   - Image normalization and alignment
   - Quality assessment and filtering

3. **Change Detection**
   - Pixel-level comparison algorithms
   - Machine learning-based classification
   - Confidence scoring for detected changes
   - Spatial analysis and pattern recognition

4. **Alert Processing**
   - Threshold-based filtering
   - Change event categorization
   - User preference matching
   - Notification queue management

### Technology Stack

- **Backend**: Python/Node.js for data processing
- **Frontend**: Web-based interface with mapping capabilities
- **Database**: Spatial database for AOI and user data storage
- **APIs**: Google Earth Engine, Bhuvan integration
- **Mapping**: Interactive web mapping libraries
- **Export**: GIS format conversion utilities

## User Workflow

1. **Account Setup**
   - User registration and authentication
   - Profile configuration

2. **AOI Definition**
   - Navigate to interactive map
   - Draw/define areas of interest
   - Save AOI with descriptive metadata

3. **Alert Configuration**
   - Select change types to monitor
   - Set confidence thresholds
   - Configure notification preferences

4. **Monitoring**
   - System automatically processes new satellite data
   - Change detection runs on scheduled intervals
   - Alerts generated based on user criteria

5. **Response**
   - Receive notifications via configured channels
   - Review change reports and visualizations
   - Export data for further analysis

## Evaluation Criteria

### Functionality (Primary)
- ✅ Multiple AOI definition and management
- ✅ User account system with AOI persistence
- ✅ Configurable confidence thresholds
- ✅ Automated change detection processing
- ✅ GIS-compatible data export

### User Experience
- ✅ Intuitive web-based interface
- ✅ No software installation required
- ✅ Interactive map-based AOI definition
- ✅ Clear alert configuration options
- ✅ Responsive design for multiple devices

### Reliability
- ✅ Robust change detection algorithms
- ✅ False positive minimization
- ✅ Cloud and shadow masking
- ✅ High detection accuracy
- ✅ Minimal false negatives

### Scalability & Flexibility
- ✅ Modular architecture for algorithm extension
- ✅ Plugin-based detection system
- ✅ Configurable processing parameters
- ✅ Support for new alert types
- ✅ Horizontal scaling capabilities

## Expected Outcomes

1. **Fully Automated System**
   - End-to-end automation from data acquisition to alert delivery
   - Minimal manual intervention required
   - Scheduled processing and monitoring

2. **Cloud-Integrated Processing**
   - Advanced cloud and shadow masking
   - Weather-aware change detection
   - Quality-based image filtering

3. **User-Friendly Platform**
   - Accessible via web browser
   - Intuitive interface design
   - Comprehensive documentation and help

4. **Exportable Results**
   - GeoJSON format support
   - Shapefile compatibility
   - Standard GIS format integration

## Installation & Setup

```bash
# Clone repository
git clone https://github.com/param2610-cloud/bhooDrishti.git

# Install dependencies
cd bhooDrishti
npm install  # or pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and database settings

# Start development server
npm start  # or python app.py
```

## Configuration

### Environment Variables
```
BHUVAN_API_KEY=your_bhuvan_api_key
GOOGLE_EARTH_ENGINE_KEY=your_gee_key
DATABASE_URL=your_database_connection
NOTIFICATION_SERVICE_KEY=your_notification_key
```

### Alert Types Configuration
```json
{
  "alertTypes": [
    "deforestation",
    "building_construction",
    "water_reclamation",
    "land_occupation"
  ],
  "confidenceThresholds": {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.9
  }
}
```

## API Documentation

### AOI Management
- `POST /api/aoi` - Create new AOI
- `GET /api/aoi` - List user AOIs
- `PUT /api/aoi/:id` - Update AOI
- `DELETE /api/aoi/:id` - Delete AOI

### Change Detection
- `GET /api/changes/:aoiId` - Get changes for AOI
- `POST /api/detection/trigger` - Manual detection trigger
- `GET /api/detection/status` - Check processing status

### Export
- `GET /api/export/:aoiId/geojson` - Export as GeoJSON
- `GET /api/export/:aoiId/shapefile` - Export as Shapefile

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Support

For questions and support, please contact:
- Email: support@bhooDrishti.com
- Documentation: [Wiki](https://github.com/your-org/bhooDrishti/wiki)
- Issues: [GitHub Issues](https://github.com/your-org/bhooDrishti/issues)

## Acknowledgments

- Indian Space Research Organisation (ISRO) for Bhuvan data access
- Google Earth Engine for satellite imagery APIs
- Open source GIS community for mapping libraries
