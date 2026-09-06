export type Period = 'Morning' | 'Afternoon' | 'Evening' | 'Night' | 'Unknown'
export interface Delivery { id:string; age:number|null; rating:number|null; storeLat:number; storeLng:number; dropLat:number; dropLng:number; date:Date; dateKey:string; orderTime:string; weather:string; traffic:string; vehicle:string; area:string; category:string; deliveryTime:number; distance:number; period:Period; hour:number; day:string; month:string }
export interface Filters { start:string; end:string; weather:string; traffic:string; vehicle:string; area:string; category:string; period:string }
