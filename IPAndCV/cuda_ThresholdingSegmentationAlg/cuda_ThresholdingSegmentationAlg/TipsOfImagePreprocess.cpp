




//int nx = inputsize[0];
//int ny = inputsize[1]; 
//int nz = inputsize[2];
//int nxy = inputsize[0]*inputsize[1];
//int size = inputsize[0] * inputsize[1] * inputsize[2];
//
//std::vector<std::vector<int>>  neighors;
//neighors.clear();
////try
////{
//neighors.resize(0x40);
//for ( int k = 0; k < neighors.size(); k++ )
//{
//	if ( ! ( k & 0x1 ) ) neighors[ k ].push_back( -1 );
//	if ( ! ( k & 0x2 ) ) neighors[ k ].push_back( 1 );
//
//	if ( ! ( k & 0x4 ) ) neighors[ k ].push_back( -nx );
//	if ( ! ( k & 0x8 ) ) neighors[ k ].push_back( nx );
//
//	if ( ! ( k & 0x10 ) ) neighors[ k ].push_back( -nxy );
//	if ( ! ( k & 0x20 ) ) neighors[ k ].push_back( nxy );
//
//}


// eight neighbor region
//int neighbor[8];
// only for eight region, full
//neighbor[0] = -1;
//neighbor[1] = 1;

//neighbor[2] = -nx;
//neighbor[3] = nx;

//neighbor[4] = nx - 1;
//neighbor[5] = nx + 1;

//neighbor[6] = -nx - 1;
//neighbor[7] = -nx + 1;