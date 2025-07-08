import React from 'react';
import { 
  MapPin, 
  Bell, 
  AlertTriangle, 
  CheckCircle, 
  Download, 
  Eye,
  Waves,
  Trees,
  Building,
  Satellite,
  TrendingUp,
} from 'lucide-react';

const Dashboard = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Quick Overview */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-gray-900 mb-6">Quick Overview</h1>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Total AOIs */}
            <div className="bg-blue-50 rounded-lg p-6 border border-blue-100">
              <div className="flex flex-col items-center text-center">
                <MapPin className="h-8 w-8 text-blue-600 mb-3" />
                <div className="text-3xl font-bold text-blue-600 mb-1">3</div>
                <div className="text-sm text-gray-600">Total AOIs</div>
                <div className="text-xs text-gray-500 mt-1">2 Active</div>
              </div>
            </div>

            {/* Total Alerts */}
            <div className="bg-orange-50 rounded-lg p-6 border border-orange-100">
              <div className="flex flex-col items-center text-center">
                <Bell className="h-8 w-8 text-orange-600 mb-3" />
                <div className="text-3xl font-bold text-orange-600 mb-1">3</div>
                <div className="text-sm text-gray-600">Total Alerts</div>
                <div className="text-xs text-gray-500 mt-1">2 New Alerts</div>
              </div>
            </div>

            {/* High Priority Alerts */}
            <div className="bg-red-50 rounded-lg p-6 border border-red-100">
              <div className="flex flex-col items-center text-center">
                <AlertTriangle className="h-8 w-8 text-red-600 mb-3" />
                <div className="text-3xl font-bold text-red-600 mb-1">3</div>
                <div className="text-sm text-gray-600">High Priority Alerts</div>
                <div className="text-xs text-gray-500 mt-1">Requires Attention</div>
              </div>
            </div>

            {/* System Status */}
            <div className="bg-green-50 rounded-lg p-6 border border-green-100">
              <div className="flex flex-col items-center text-center">
                <CheckCircle className="h-8 w-8 text-green-600 mb-3" />
                <div className="text-3xl font-bold text-green-600 mb-1">90%</div>
                <div className="text-sm text-gray-600">System Status</div>
                <div className="text-xs text-gray-500 mt-1">Uptime</div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - System Processing Status */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow p-6 mb-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-2">System Processing Status</h2>
              <p className="text-sm text-gray-600 mb-6">Real-time status of satellite imagery processing and change detection</p>
              
              <div className="space-y-6">
                {/* Imagery Download */}
                <div className="text-center">
                  <Satellite className="h-8 w-8 text-blue-600 mx-auto mb-2" />
                  <div className="text-sm font-medium text-gray-900">Imagery Download</div>
                  <div className="text-xs text-gray-500 mb-2">Latest satellite data</div>
                  <div className="flex items-center justify-center space-x-1">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span className="text-sm text-green-600">Up to date</span>
                  </div>
                </div>

                {/* Change Detection */}
                <div className="text-center">
                  <TrendingUp className="h-8 w-8 text-orange-600 mx-auto mb-2" />
                  <div className="text-sm font-medium text-gray-900">Change Detection</div>
                  <div className="text-xs text-gray-500 mb-2">Processing queue</div>
                  <div className="flex items-center justify-center space-x-1">
                    <div className="h-2 w-2 bg-blue-500 rounded-full animate-pulse"></div>
                    <span className="text-sm text-blue-600">Processing</span>
                  </div>
                </div>

                {/* Alert System */}
                <div className="text-center">
                  <Bell className="h-8 w-8 text-green-600 mx-auto mb-2" />
                  <div className="text-sm font-medium text-gray-900">Alert System</div>
                  <div className="text-xs text-gray-500 mb-2">Notification service</div>
                  <div className="flex items-center justify-center space-x-1">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span className="text-sm text-green-600">Active</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column - Recent Alerts and AOI Status */}
          <div className="lg:col-span-2">
            {/* Recent Alerts */}
            <div className="bg-white rounded-lg shadow p-6 mb-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-semibold text-gray-900">Recent Alerts</h2>
                <button className="text-sm text-blue-600 hover:text-blue-800">View All</button>
              </div>
              <p className="text-sm text-gray-600 mb-4">Latest change detection alerts from your monitored areas</p>
              
              <div className="space-y-4">
                {/* Forest Reserve Alert */}
                <div className="flex items-center justify-between p-3 bg-red-50 rounded-lg border border-red-100">
                  <div className="flex items-center space-x-3">
                    <Trees className="h-5 w-5 text-green-600" />
                    <div>
                      <div className="font-medium text-gray-900">Forest Reserve Area</div>
                      <div className="text-sm text-gray-600">Deforestation Detected</div>
                      <div className="text-xs text-gray-500">1/6/2025, 4:00:00 PM</div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="px-2 py-1 text-xs font-medium bg-red-100 text-red-800 rounded">high</span>
                    <span className="text-sm text-gray-600">87% confidence</span>
                    <Eye className="h-4 w-4 text-gray-400" />
                    <Download className="h-4 w-4 text-gray-400" />
                  </div>
                </div>

                {/* Lake Monitoring Alert */}
                <div className="flex items-center justify-between p-3 bg-yellow-50 rounded-lg border border-yellow-100">
                  <div className="flex items-center space-x-3">
                    <Waves className="h-5 w-5 text-blue-600" />
                    <div>
                      <div className="font-medium text-gray-900">Lake Monitoring Zone</div>
                      <div className="text-sm text-gray-600">Water Body_change Detected</div>
                      <div className="text-xs text-gray-500">1/6/2025, 1:45:00 PM</div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="px-2 py-1 text-xs font-medium bg-yellow-100 text-yellow-800 rounded">medium</span>
                    <span className="text-sm text-gray-600">72% confidence</span>
                    <Eye className="h-4 w-4 text-gray-400" />
                    <Download className="h-4 w-4 text-gray-400" />
                  </div>
                </div>

                {/* Urban Development Alert */}
                <div className="flex items-center justify-between p-3 bg-red-50 rounded-lg border border-red-100">
                  <div className="flex items-center space-x-3">
                    <Building className="h-5 w-5 text-gray-600" />
                    <div>
                      <div className="font-medium text-gray-900">Urban Development Area</div>
                      <div className="text-sm text-gray-600">Illegal Construction Detected</div>
                      <div className="text-xs text-gray-500">1/6/2025, 10:15:00 PM</div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="px-2 py-1 text-xs font-medium bg-red-100 text-red-800 rounded">high</span>
                    <span className="text-sm text-gray-600">94% confidence</span>
                    <Eye className="h-4 w-4 text-gray-400" />
                    <Download className="h-4 w-4 text-gray-400" />
                  </div>
                </div>
              </div>
            </div>

            {/* AOI Status */}
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-semibold text-gray-900">AOI Status</h2>
                <button className="text-sm text-blue-600 hover:text-blue-800">Manage AOIs</button>
              </div>
              <p className="text-sm text-gray-600 mb-6">Overview of your monitored Areas of Interest</p>
              
              <div className="space-y-6">
                {/* Forest Reserve Area */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <Trees className="h-4 w-4 text-green-600" />
                      <span className="font-medium text-gray-900">Forest Reserve Area</span>
                    </div>
                    <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded">active</span>
                  </div>
                  <div className="flex items-center space-x-4 text-sm text-gray-600 mb-2">
                    <span className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                      <span>deforestation</span>
                    </span>
                    <span className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                      <span>illegal construction</span>
                    </span>
                  </div>
                  <div className="mb-2">
                    <div className="flex justify-between text-sm mb-1">
                      <span>Confidence Threshold</span>
                      <span>80%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div className="bg-blue-600 h-2 rounded-full" style={{width: '80%'}}></div>
                    </div>
                  </div>
                  <div className="flex justify-between text-sm text-gray-500">
                    <span>3 alerts</span>
                    <span>1/6/2025, 4:00:00 PM</span>
                  </div>
                </div>

                {/* Lake Monitoring Zone */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <Waves className="h-4 w-4 text-blue-600" />
                      <span className="font-medium text-gray-900">Lake Monitoring Zone</span>
                    </div>
                    <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded">active</span>
                  </div>
                  <div className="flex items-center space-x-4 text-sm text-gray-600 mb-2">
                    <span className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                      <span>water body_change</span>
                    </span>
                    <span className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                      <span>encroachment</span>
                    </span>
                  </div>
                  <div className="mb-2">
                    <div className="flex justify-between text-sm mb-1">
                      <span>Confidence Threshold</span>
                      <span>60%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div className="bg-blue-600 h-2 rounded-full" style={{width: '60%'}}></div>
                    </div>
                  </div>
                  <div className="flex justify-between text-sm text-gray-500">
                    <span>1 alerts</span>
                    <span>1/6/2025, 1:45:00 PM</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;